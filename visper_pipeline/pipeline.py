
import sys
import json
import cv2

from pathlib import Path, PureWindowsPath
from argparse import Namespace

# ============================================================
# IMPORTS DEL PROYECTO
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PREPARE_DIR = PROJECT_DIR / "data_prepare"

sys.path.insert(0, str(DATA_PREPARE_DIR))

from download import load_ids, download_batch
from filter_metadata import filter_metadata
from crop_videos import main as crop_batch


# ============================================================
# CONFIGURACIÓN
# ============================================================

IDS_FILE = (
    PROJECT_DIR.parent
    / "preselect"
    / "metadata"
    / "extracted"
    / "Spanish"
    / "Train"
    / "videos_ids_train.txt"
)

VIDEO_DIR = Path(r"D:\ViSpeR\videos")
ANNOTATIONS_DIR = Path(r"D:\ViSpeR\metadata_20")
CLIPS_DIR = Path(r"D:\ViSpeR\clips")

GLOBAL_METADATA = CLIPS_DIR / "metadata.json"

BATCH_SIZE = 20
DOWNLOAD_WORKERS = 3
EXPECTED_FPS = 25
EXPECTED_WIDTH = 96
EXPECTED_HEIGHT = 96

CROP_ARGS = Namespace(
    save_path=str(CLIPS_DIR),
    json_path=str(ANNOTATIONS_DIR),
    video_dir=str(VIDEO_DIR),
    save_audio=False,
    merge_audio=False,
    use_ffmpeg=False
)


# ============================================================
# UTILIDADES
# ============================================================

def normalize_path(path):
    """Normaliza rutas Windows para comparar claves del metadata."""
    return str(PureWindowsPath(str(path))).casefold()


def load_global_metadata():
    if not GLOBAL_METADATA.exists():
        return {}

    with open(GLOBAL_METADATA, "r", encoding="utf-8") as f:
        return json.load(f)


def load_annotations(video_id):
    """Carga las anotaciones de un video."""
    annotation_file = ANNOTATIONS_DIR / f"{video_id}.json"

    if not annotation_file.exists():
        return None

    with open(annotation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get(video_id)


# ============================================================
# DETECCIÓN DE VIDEOS COMPLETADOS
# ============================================================

def get_completed_ids():
    """
    Un video se considera completo cuando:
    - Tiene anotaciones.
    - Existe cada clip esperado.
    - Cada clip está registrado en el metadata global.
    """

    global_metadata = load_global_metadata()

    registered_paths = {
        normalize_path(path)
        for path in global_metadata
    }

    completed = set()

    for annotation_file in ANNOTATIONS_DIR.glob("*.json"):

        video_id = annotation_file.stem

        with open(annotation_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        segments = data.get(video_id, [])

        if not segments:
            continue

        all_valid = True

        for index, segment in enumerate(segments):

            clip_path = (
                CLIPS_DIR
                / "videos"
                / video_id
                / f"{index:05d}.mp4"
            )

            if not clip_path.is_file():
                all_valid = False
                break

            if normalize_path(clip_path) not in registered_paths:
                all_valid = False
                break

        if all_valid:
            completed.add(video_id)

    return completed


def select_next_batch():

    all_ids = load_ids(IDS_FILE)
    completed_ids = get_completed_ids()

    pending_ids = [
        video_id
        for video_id in all_ids
        if video_id not in completed_ids
    ]

    batch_ids = pending_ids[:BATCH_SIZE]

    print("\n========== SELECCIÓN ==========")
    print(f"Videos en la lista: {len(all_ids)}")
    print(f"Videos completados: {len(completed_ids)}")
    print(f"Videos pendientes: {len(pending_ids)}")
    print(f"Tanda seleccionada: {len(batch_ids)}")

    return batch_ids


# ============================================================
# VALIDACIÓN DE CLIPS
# ============================================================

def validate_clip(clip_path, expected_frames):
    """
    Comprueba que el clip:
    - Se pueda abrir.
    - Tenga resolución 96x96.
    - Tenga FPS 25.
    - Tenga la cantidad esperada de frames.
    """

    cap = cv2.VideoCapture(str(clip_path))

    if not cap.isOpened():
        cap.release()
        return False, "No se pudo abrir"

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    if width != EXPECTED_WIDTH or height != EXPECTED_HEIGHT:
        return False, f"Resolución incorrecta: {width}x{height}"

    if abs(fps - EXPECTED_FPS) > 0.1:
        return False, f"FPS incorrectos: {fps}"

    if frames != expected_frames:
        return False, (
            f"Frames incorrectos: {frames}, "
            f"esperados: {expected_frames}"
        )

    return True, "OK"


def validate_video(video_id, global_metadata):
    """
    Verifica todos los clips de un video contra sus anotaciones
    y el metadata global.
    """

    segments = load_annotations(video_id)

    if segments is None or len(segments) == 0:
        return False, "No hay anotaciones"

    for index, segment in enumerate(segments):

        clip_path = (
            CLIPS_DIR
            / "videos"
            / video_id
            / f"{index:05d}.mp4"
        )

        if not clip_path.is_file():
            return False, f"Falta el clip {index:05d}"

        metadata_key = normalize_path(clip_path)

        matching_key = next(
            (
                key for key in global_metadata
                if normalize_path(key) == metadata_key
            ),
            None
        )

        if matching_key is None:
            return False, f"Clip {index:05d} ausente del metadata global"

        expected_frames = len(segment["landmarks"])

        metadata_entry = global_metadata[matching_key]

        if metadata_entry.get("length") != expected_frames:
            return False, f"Longitud incorrecta en metadata: {index:05d}"

        valid, message = validate_clip(
            clip_path,
            expected_frames
        )

        if not valid:
            return False, f"Clip {index:05d}: {message}"

    return True, f"{len(segments)} clips validados"


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def run_pipeline():

    batch_ids = select_next_batch()

    if not batch_ids:
        print("\nNo quedan videos pendientes.")
        return

    print("\n========== DESCARGA ==========")

    ready_ids, download_failed = download_batch(
        batch_ids,
        workers=DOWNLOAD_WORKERS
    )

    ready_ids = list(ready_ids)

    print(f"\nVideos disponibles: {len(ready_ids)}")
    print(f"Descargas fallidas: {len(download_failed)}")

    if not ready_ids:
        print("No hay videos disponibles para continuar.")
        return

    # --------------------------------------------------------
    # EXTRACCIÓN DE ANOTACIONES
    # --------------------------------------------------------

    print("\n========== ANOTACIONES ==========")

    found_ids, missing_ids = filter_metadata(ready_ids)

    found_ids = set(found_ids)
    missing_ids = set(missing_ids)

    print(f"\nAnotaciones encontradas: {len(found_ids)}")
    print(f"Anotaciones faltantes: {len(missing_ids)}")

    crop_ids = [
        video_id
        for video_id in ready_ids
        if video_id in found_ids
    ]

    if not crop_ids:
        print("No hay videos con anotaciones para procesar.")
        return

    # --------------------------------------------------------
    # GENERACIÓN DE CLIPS
    # --------------------------------------------------------

    print("\n========== GENERACIÓN DE CLIPS ==========")

    try:
        crop_result = crop_batch(
            CROP_ARGS,
            batch_ids=crop_ids
        )

        print("\nResultado del procesamiento:")
        print(crop_result)

    except Exception as error:
        print("\nERROR durante el procesamiento de clips:")
        print(repr(error))
        print("Se conservan los videos originales.")
        return

    # --------------------------------------------------------
    # VALIDACIÓN
    # --------------------------------------------------------

    print("\n========== VALIDACIÓN ==========")

    global_metadata = load_global_metadata()

    validated_ids = []
    invalid_ids = []

    for video_id in crop_ids:

        valid, message = validate_video(
            video_id,
            global_metadata
        )

        if valid:
            validated_ids.append(video_id)
            print(f"[OK] {video_id}: {message}")
        else:
            invalid_ids.append(video_id)
            print(f"[ERROR] {video_id}: {message}")

    # --------------------------------------------------------
    # LIMPIEZA SEGURA DE ORIGINALES
    # --------------------------------------------------------

    print("\n========== LIMPIEZA ==========")

    deleted = 0

    # Ruta absoluta y resuelta de la carpeta autorizada.
    safe_video_dir = VIDEO_DIR.resolve()

    for video_id in validated_ids:

        original_path = (VIDEO_DIR / f"{video_id}.mp4").resolve()

        # 1. Verificar que el archivo esté dentro de VIDEO_DIR.
        if original_path.parent != safe_video_dir:
            print(f"[BLOQUEADO] Ruta fuera del directorio autorizado: {original_path}")
            continue

        # 2. Verificar que sea un archivo, no un directorio.
        if not original_path.is_file():
            print(f"[NO EXISTE] {original_path}")
            continue

        # 3. Verificar extensión y nombre esperado.
        if original_path.suffix.lower() != ".mp4":
            print(f"[BLOQUEADO] Extensión inesperada: {original_path}")
            continue

        if original_path.stem != video_id:
            print(f"[BLOQUEADO] Nombre inesperado: {original_path}")
            continue

        # 4. Eliminar exclusivamente el original validado.
        original_path.unlink()

        deleted += 1
        print(f"[ELIMINADO] {original_path}")

        print(f"\nOriginales eliminados: {deleted}")

    # --------------------------------------------------------
    # RESUMEN
    # --------------------------------------------------------

    print("\n========== RESUMEN DE TANDA ==========")
    print(f"Seleccionados: {len(batch_ids)}")
    print(f"Descargados/disponibles: {len(ready_ids)}")
    print(f"Sin anotaciones: {len(missing_ids)}")
    print(f"Validados: {len(validated_ids)}")
    print(f"Con errores de validación: {len(invalid_ids)}")
    print(f"Originales eliminados: {deleted}")

    if download_failed:
        print("\nDescargas fallidas:")
        print(download_failed)

    if missing_ids:
        print("\nSin anotaciones:")
        print(sorted(missing_ids))

    if invalid_ids:
        print("\nVideos que necesitan revisión:")
        print(invalid_ids)

    print("\nTanda finalizada.")


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    run_pipeline()
