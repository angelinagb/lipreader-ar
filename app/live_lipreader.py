"""
    cámara -> bbox de la boca -> recorte 96x96 RGB a 25 fps -> detección de frases por
    movimiento de labios -> modelo (CTC) -> texto

No sirve una ventana fija de pocos frames. Acá se detecta cuándo empieza y termina cada frase mirando cuánto se mueven
los labios, y se corre el modelo sobre la frase entera.

Instalación:
    pip install opencv-python mediapipe numpy torch

Uso:
    python live_lipreader.py --ckpt ../checkpoints/best_cer.pth
    python live_lipreader.py --ckpt ../checkpoints/best_cer.pth --show      # con preview
    python live_lipreader.py --ckpt ../checkpoints/best_cer.pth --beam      # beam search + LM (tu corrector)

Con --show, la tecla 'q' sale. Sin --show corre sin ventanas (Ctrl+C para salir).
"""
import argparse
import queue
import sys
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from lipreader_model import Predictor

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = Path(__file__).with_name("face_landmarker.task")

# Landmarks de los labios del Face Mesh (contorno externo + interno)
LIPS_IDX = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
]


# ----------------------------------------------------------------------------
# Geometría de la boca
# ----------------------------------------------------------------------------
def ensure_model():
    if not MODEL_PATH.exists():
        print("Descargando modelo face_landmarker.task ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def mouth_box(landmarks, w, h, margin):
    """Cuadrado alrededor de la boca, como array [cx, cy, lado] en píxeles."""
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in LIPS_IDX])
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + 2 * margin)
    return np.array([cx, cy, side])


def box_to_xyxy(box, w, h):
    """Pasa [cx, cy, lado] a (x1, y1, x2, y2) enteros, sin salirse de la imagen."""
    cx, cy, side = box
    s = int(round(min(side, w, h)))
    x1 = int(np.clip(round(cx - s / 2), 0, w - s))
    y1 = int(np.clip(round(cy - s / 2), 0, h - s))
    return x1, y1, x1 + s, y1 + s


def lip_aperture(landmarks, w, h):
    """Apertura interna de la boca (labio sup. 13 / inf. 14) sobre el ancho de la boca (61-291)."""
    def p(i):
        return np.array([landmarks[i].x * w, landmarks[i].y * h])
    width = np.linalg.norm(p(61) - p(291))
    return float(np.linalg.norm(p(13) - p(14)) / max(width, 1e-6))


# ----------------------------------------------------------------------------
# Segmentación de frases por actividad de labios
# ----------------------------------------------------------------------------
class UtteranceSegmenter:
    """
    Recibe un frame (25 fps) y la apertura de la boca. "Actividad" = cuánto varió la apertura
    en los últimos `act_win` frames. Cuando hay actividad empieza una frase; cuando pasan `hang`
    frames sin actividad, termina y devuelve el clip (np.uint8, (T, 96, 96, 3)).
    """

    def __init__(self, thr=0.03, hang=12, preroll=6, tail=4, min_frames=20, max_frames=150, act_win=6):
        self.thr, self.hang, self.tail = thr, hang, tail
        self.min_frames, self.max_frames = min_frames, max_frames
        self.aps = deque(maxlen=act_win)
        self.pre = deque(maxlen=preroll)   # frames previos al inicio, para no cortar la primera sílaba
        self.clip = []
        self.speaking = False
        self.quiet = 0
        self.activity = 0.0

    def reset(self):
        self.aps.clear()
        self.pre.clear()
        self.clip = []
        self.speaking = False
        self.quiet = 0
        self.activity = 0.0

    def update(self, frame_rgb, aperture):
        self.aps.append(aperture)
        self.activity = (max(self.aps) - min(self.aps)) if len(self.aps) == self.aps.maxlen else 0.0
        active = self.activity > self.thr

        if not self.speaking:
            self.pre.append(frame_rgb)
            if active:
                self.speaking = True
                self.clip = list(self.pre)
                self.quiet = 0
            return None

        self.clip.append(frame_rgb)
        self.quiet = 0 if active else self.quiet + 1
        if self.quiet >= self.hang or len(self.clip) >= self.max_frames:
            return self._finish()
        return None

    def _finish(self):
        drop = max(0, self.quiet - self.tail)           # saco el silencio sobrante del final
        clip = self.clip[: len(self.clip) - drop] if drop else self.clip
        self.pre.clear()
        self.clip = []
        self.speaking = False
        self.quiet = 0
        return np.stack(clip) if len(clip) >= self.min_frames else None


# ----------------------------------------------------------------------------
# Hilo de captura + recorte + segmentación
# ----------------------------------------------------------------------------
class MouthStream(threading.Thread):
    def __init__(self, segmenter, out_q, camera=0, size=96, margin=0.25, smooth=0.6,
                 fps=25.0, lost_tolerance=5, draw=False):
        super().__init__(daemon=True, name="mouth-stream")
        self.seg, self.out_q = segmenter, out_q
        self.camera, self.size, self.margin, self.smooth = camera, size, margin, smooth
        self.fps, self.lost_tolerance, self.draw = fps, lost_tolerance, draw
        self.stop_evt = threading.Event()
        self.error = None
        self.preview = None   # (frame con overlay, recorte BGR) para --show

    def emit(self, clip):
        try:
            self.out_q.put_nowait(clip)
        except queue.Full:                      # el modelo no da abasto: descarto la frase más vieja
            try:
                self.out_q.get_nowait()
            except queue.Empty:
                pass
            self.out_q.put_nowait(clip)

    def run(self):
        cap = cv2.VideoCapture(self.camera)
        if not cap.isOpened():
            self.error = f"No pude abrir la cámara {self.camera}"
            self.stop_evt.set()
            return
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
        )

        min_dt = 1.0 / self.fps
        next_push = 0.0
        smoothed = None
        last_ap = 0.0
        lost = 0
        last_ts = -1

        try:
            with vision.FaceLandmarker.create_from_options(options) as landmarker:
                while not self.stop_evt.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        self.error = "No se pudo leer un frame de la cámara"
                        break
                    now = time.monotonic()
                    h, w = frame.shape[:2]

                    rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_full)
                    ts = max(int(now * 1000), last_ts + 1)
                    last_ts = ts
                    result = landmarker.detect_for_video(mp_image, ts)

                    if result.face_landmarks:
                        lm = result.face_landmarks[0]
                        lost = 0
                        box = mouth_box(lm, w, h, self.margin)
                        smoothed = box if smoothed is None else (
                            self.smooth * smoothed + (1 - self.smooth) * box
                        )
                        last_ap = lip_aperture(lm, w, h)
                    else:
                        lost += 1
                        if lost > self.lost_tolerance:
                            smoothed = None
                            self.seg.reset()    # cara perdida: descarto la frase en curso

                    crop_bgr = None
                    if smoothed is not None:
                        x1, y1, x2, y2 = box_to_xyxy(smoothed, w, h)
                        roi = frame[y1:y2, x1:x2]
                        interp = cv2.INTER_AREA if (x2 - x1) > self.size else cv2.INTER_CUBIC
                        crop_bgr = cv2.resize(roi, (self.size, self.size), interpolation=interp)

                        # Remuestreo a 25 fps (el fps del entrenamiento)
                        if now >= next_push:
                            clip = self.seg.update(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB), last_ap)
                            if clip is not None:
                                self.emit(clip)
                            next_push += min_dt
                            if next_push < now:
                                next_push = now + min_dt

                        if self.draw:
                            state = "HABLANDO" if self.seg.speaking else "silencio"
                            color = (0, 255, 0) if self.seg.speaking else (200, 200, 200)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(
                                frame,
                                f"{state} | actividad {self.seg.activity:.3f} (umbral {self.seg.thr})"
                                f" | frames {len(self.seg.clip)}",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                    elif self.draw:
                        cv2.putText(frame, "No se detecta cara", (10, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    if self.draw:
                        self.preview = (frame, crop_bgr)
        finally:
            cap.release()
            self.stop_evt.set()


# ----------------------------------------------------------------------------
# Hilo de inferencia
# ----------------------------------------------------------------------------
def inference_loop(stream, predictor, decode, out_q, fps):
    while not stream.stop_evt.is_set():
        try:
            clip = out_q.get(timeout=0.3)
        except queue.Empty:
            continue
        t0 = time.perf_counter()
        lp = predictor.log_probs(clip)
        text = decode(lp)
        ms = (time.perf_counter() - t0) * 1000
        print(f"[{time.strftime('%H:%M:%S')}] {text or '(vacío)'!r}   "
              f"({len(clip)} frames = {len(clip) / fps:.1f} s, inferencia {ms:.0f} ms)", flush=True)


def find_repo(start):
    """Sube carpetas hasta encontrar la que tiene `corrector/` (igual que el notebook)."""
    p = Path(start).resolve()
    while not (p / "corrector").is_dir() and p != p.parent:
        p = p.parent
    return p if (p / "corrector").is_dir() else None


def load_beam_decoder(predictor, repo):
    """Beam search + LM con tu paquete `corrector`, con la misma config del notebook."""
    repo = Path(repo) if repo else find_repo(Path.cwd()) or find_repo(Path(__file__).parent)
    if repo is None:
        raise SystemExit("No encontré la carpeta `corrector/`. Pasala con --repo")
    sys.path.insert(0, str(repo))
    from corrector.beam import beam_ctc, prefijos_de
    from corrector.datos import cargar_splits
    from corrector.lenguaje import ModeloLenguaje

    print("Construyendo modelo de lenguaje para beam search ...")
    lm = ModeloLenguaje(cargar_splits()["train"]["label"])
    pre = prefijos_de(lm.vocab)
    cfg = dict(alfa=0.3, beta=2.0, B=32)

    def decode(lp):
        return beam_ctc(lp.astype(np.float32), predictor.idx_to_char, predictor.blank,
                        lm=lm, prefijos=pre, **cfg)
    return decode


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint, ej. ../checkpoints/best_cer.pth")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--rnn", default="gru", choices=["gru", "lstm"])
    ap.add_argument("--margin", type=float, default=0.25,
                    help="margen alrededor de la boca; ajustalo para que el encuadre se parezca al de tus clips")
    ap.add_argument("--smooth", type=float, default=0.6, help="suavizado del box (0 = nada)")
    ap.add_argument("--fps", type=float, default=25.0, help="fps del entrenamiento")
    ap.add_argument("--thr", type=float, default=0.03,
                    help="umbral de actividad de labios para detectar habla (miralo con --show)")
    ap.add_argument("--hang", type=int, default=12, help="frames sin actividad para cerrar la frase")
    ap.add_argument("--min-frames", type=int, default=20, help="frases más cortas se descartan")
    ap.add_argument("--max-frames", type=int, default=150, help="largo máximo de una frase (se corta ahí)")
    ap.add_argument("--beam", action="store_true", help="beam search + LM (requiere tu paquete corrector)")
    ap.add_argument("--repo", default=None, help="raíz del repo que contiene corrector/")
    ap.add_argument("--show", action="store_true", help="mostrar preview")
    args = ap.parse_args()

    ensure_model()

    predictor = Predictor(args.ckpt, rnn_type=args.rnn)
    predictor.warmup()
    print(f"Modelo cargado en {predictor.device} | {predictor.info}")

    decode = load_beam_decoder(predictor, args.repo) if args.beam else predictor.greedy

    out_q = queue.Queue(maxsize=4)
    seg = UtteranceSegmenter(thr=args.thr, hang=args.hang, min_frames=args.min_frames,
                             max_frames=args.max_frames)
    stream = MouthStream(seg, out_q, camera=args.camera, margin=args.margin,
                         smooth=args.smooth, fps=args.fps, draw=args.show)
    stream.start()
    threading.Thread(target=inference_loop, args=(stream, predictor, decode, out_q, args.fps),
                     daemon=True, name="inference").start()

    print("Escuchando con los ojos. Hablá moviendo los labios; Ctrl+C para salir"
          + (" (o 'q' en la ventana)." if args.show else "."))
    try:
        while not stream.stop_evt.is_set():
            if args.show:
                if stream.preview is not None:
                    frame, crop = stream.preview
                    cv2.imshow("Camara", frame)
                    if crop is not None:
                        cv2.imshow("Boca 96x96", cv2.resize(crop, None, fx=4, fy=4,
                                                            interpolation=cv2.INTER_NEAREST))
                if cv2.waitKey(15) & 0xFF in (ord("q"), 27):
                    break
            else:
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_evt.set()
        stream.join(timeout=2)
        if args.show:
            cv2.destroyAllWindows()

    if stream.error:
        raise SystemExit(stream.error)


if __name__ == "__main__":
    main()