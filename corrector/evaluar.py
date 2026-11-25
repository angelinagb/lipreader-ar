"""
Evalua el corrector: CER y WER antes y después de corregir

Reglas: 
El diccionario se arma SOLO con train
Los ajustes (pesos, costos) se eligen mirando VAL
TEST se mira al final, una vez
"""

import argparse
import itertools
import random
import time
from pathlib import Path

import pandas as pd

from corrector.confusion import contar_pares
from .texto import normalize_text, reducir
from .corrector import Corrector
from .datos import PARQUET_POR_DEFECTO, cargar_splits
from .distancia import COSTOS_PLANOS, Costos, costos_aprendidos
from .lenguaje import ModeloLenguaje
from .metricas import cer_wer
from .ruido import romper
from .texto import normalize_text

RESULTADOS = Path(__file__).resolve().parent / "resultados"


def configuraciones(lm):
    #Las variantes a comparar, de la más simple a la más completa.
    return {
        "letras + bigramas": Corrector(lm, peso_contexto=0.3),
        "fonemas + bigramas": Corrector(lm, peso_contexto=0.3, comparar="fonemas"),
        "combinada 0.5 + bigramas": Corrector(lm, peso_contexto=0.3, comparar="combinada", alfa=0.5),
        "combinada 0.7 + bigramas": Corrector(lm, peso_contexto=0.3, comparar="combinada", alfa=0.7),
        "letras + bigramas + protege (1.0)": Corrector(lm, peso_contexto=0.3, costo_desconocida=1.0),
        "letras + bigramas + protege (1.5)": Corrector(lm, peso_contexto=0.3, costo_desconocida=1.5),
        "letras + bigramas + protege (2.0)": Corrector(lm, peso_contexto=0.3, costo_desconocida=2.0),
    }


def comparar(referencias, rotas, correctores):
    #Una fila por configuración con CER/WER antes y después
    cer0, wer0 = cer_wer(referencias, rotas)
    filas = [{"configuración": "sin corregir", "CER": cer0, "WER": wer0, "seg": 0.0}]
    for nombre, corr in correctores.items():
        t = time.time()
        corregidas = [corr.corregir(r) for r in rotas]
        cer, wer = cer_wer(referencias, corregidas)
        filas.append({"configuración": nombre, "CER": cer, "WER": wer,
                      "seg": round(time.time() - t, 1)})
    return pd.DataFrame(filas)


def mostrar(df, titulo):
    print(f"\n=== {titulo} ===")
    print(df.to_string(index=False, formatters={"CER": "{:.1%}".format, "WER": "{:.1%}".format}))


def ejemplos(referencias, rotas, corrector, n=5, seed=0):
    rng = random.Random(seed)
    print("\nEjemplos (real / rota / corregida):")
    for i in rng.sample(range(len(referencias)), min(n, len(referencias))):
        print(f"  REAL: {referencias[i]}\n  ROTA: {rotas[i]}\n  CORR: {corrector.corregir(rotas[i])}\n")


def ajustar(lm, val_frases, tasa, modo):
    #Prueba combinaciones de pesos en VAL y muestra las mejores por WER
    rng = random.Random(1)
    rotas = [romper(f, tasa, modo, rng) for f in val_frases]
    filas = []
    for mv, pf, pc in itertools.product([0.3, 0.5, 1.0], [0.0, 0.05, 0.15], [0.0, 0.3, 0.6]):
        corr = Corrector(lm, costos=Costos(mismo_visema=mv), peso_freq=pf, peso_contexto=pc)
        cer, wer = cer_wer(val_frases, [corr.corregir(r) for r in rotas])
        filas.append({"mismo_visema": mv, "peso_freq": pf, "peso_contexto": pc,
                      "CER": cer, "WER": wer})
    df = pd.DataFrame(filas).sort_values("WER")
    mostrar(df.head(8), f"Mejores pesos en VAL (ruido {modo} {tasa:.0%})")
    return df


def main():
    ap = argparse.ArgumentParser(description="Evalúa el corrector de transcripciones.")
    ap.add_argument("--parquet", default=str(PARQUET_POR_DEFECTO))
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--n", type=int, default=800, help="frases a usar (0 = todas)")
    ap.add_argument("--tasas", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5],
                    help="niveles de error simulado")
    ap.add_argument("--modo", choices=["visema", "azar"], default="visema")
    ap.add_argument("--predicciones", help="CSV con columnas reference y prediction (salida del modelo)")
    ap.add_argument("--ajustar", action="store_true", help="buscar pesos en VAL")
    ap.add_argument("--reducido", action="store_true",
                    help="simular la salida del modelo con vocabulario reducido (sin tildes, sin h, y→i)")    
    ap.add_argument("--aprender-costos", action="store_true",
                    help="con --predicciones: aprende costos de 3/5 de los videos y mide en el resto")
    args = ap.parse_args()

    if args.ajustar:
        args.split = "val"   # los ajustes se eligen siempre mirando VAL

    RESULTADOS.mkdir(exist_ok=True)
    splits = cargar_splits(args.parquet)
    lm = ModeloLenguaje(splits["train"]["label"])
    print(f"Diccionario (train): {len(lm.vocab)} palabras distintas, {lm.total} en total")

    frases = splits[args.split]["label"].tolist()
    palabras = [p for f in frases for p in f.split()]
    fuera = sum(p not in lm for p in palabras) / len(palabras)
    print(f"Palabras de {args.split} fuera del diccionario: {fuera:.1%} "
          "(techo: esas nunca se pueden corregir)")

    if args.n:
        frases = random.Random(0).sample(frases, min(args.n, len(frases)))

    if args.ajustar:
        val = splits["val"]["label"].tolist()
        val = random.Random(0).sample(val, min(args.n or len(val), len(val)))
        ajustar(lm, val, tasa=0.3, modo=args.modo).to_csv(RESULTADOS / "ajuste_val.csv", index=False)
        return

    correctores = configuraciones(lm)

    if args.predicciones:
        pred = pd.read_csv(args.predicciones).fillna("")
        if args.aprender_costos:
            videos = sorted(pred["video_id"].unique())
            aprender = set(videos[: len(videos) * 3 // 5])
            pa = pred[pred["video_id"].isin(aprender)]
            pred = pred[~pred["video_id"].isin(aprender)]
            cs = costos_aprendidos(contar_pares(pa["reference"], pa["prediction"]))
            print(f"Costos aprendidos con {len(pa)} clips ({len(aprender)} videos) | "
                  f"se mide con {len(pred)} clips ({pred['video_id'].nunique()} videos)")
            correctores = {
                "letras + bigramas":    Corrector(lm, peso_contexto=0.3),
                "aprendida + bigramas": Corrector(lm, peso_contexto=0.3, comparar="aprendida", costos_conf=cs),
            }  
        refs = pred["reference"].astype(str).apply(normalize_text).tolist()
        hyps = pred["prediction"].astype(str).apply(normalize_text).tolist()
        df = comparar(refs, hyps, correctores)
        mostrar(df, f"Predicciones reales del modelo ({len(refs)} clips)")
        df.to_csv(RESULTADOS / "predicciones_reales.csv", index=False)
        ejemplos(refs, hyps, correctores["letras + bigramas"])
        return
    if args.reducido:
        piso = cer_wer(frases, [reducir(f) for f in frases])
        print(f"\nPiso (solo reducir, sin errores): CER {piso[0]:.1%} | WER {piso[1]:.1%}")
        correctores = {
            "reducido + bigramas": Corrector(lm, peso_contexto=0.3, reducido=True),
            "reducido + bigramas + fonemas": Corrector(lm, peso_contexto=0.3, reducido=True,
                                                       comparar="fonemas"),
        }
    todas = []
    for tasa in args.tasas:
        rng = random.Random(42)
        rotas = [romper(f, tasa, args.modo, rng) for f in frases]
        if args.reducido:
            rotas = [reducir(r) for r in rotas]
        df = comparar(frases, rotas, correctores)
        df.insert(0, "ruido", tasa)
        mostrar(df.drop(columns="ruido"), f"{args.split.upper()} · ruido {args.modo} {tasa:.0%} · {len(frases)} frases")
        todas.append(df)

    salida = RESULTADOS / f"simulado_{args.split}_{args.modo}.csv"
    pd.concat(todas).to_csv(salida, index=False)
    print(f"\nResultados guardados en {salida}")
    rotas_ej = [romper(f, args.tasas[0], args.modo, random.Random(42)) for f in frases]
    if args.reducido:
        rotas_ej = [reducir(r) for r in rotas_ej]
    ejemplos(frases, rotas_ej, next(iter(correctores.values())))


if __name__ == "__main__":
    main()
