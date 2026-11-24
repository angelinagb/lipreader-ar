"""
Modelo de lenguaje simple: cuenta palabras (unigramas) y pares de palabras
seguidas (bigramas) en las frases de TRAIN

Unigrama dice qué tan común es una palabra ("que" >> "quepa").
Bigrama dice qué tan probable es una palabra despues de otra
  ("la ola" vs "la hola"): es lo que permite usar el contexto
"""

import json
import math
from collections import Counter
from .texto import reducir

INICIO = "<s>"   # marca de comienzo de frase


class ModeloLenguaje:
    def __init__(self, frases, k=0.1):
        self.k = k   # suavizado: los pares nunca vistos no tienen probabilidad 0
        self.uni = Counter()
        self.bi = Counter()
        for frase in frases:
            palabras = frase.split()
            self.uni.update(palabras)
            self.bi.update(zip([INICIO] + palabras[:-1], palabras))
        self.total = sum(self.uni.values())
        self.vocab = sorted(self.uni)
        self.antes = Counter()   # cuántas veces aparece cada palabra como "anterior"
        for (prev, _), n in self.bi.items():
            self.antes[prev] += n
        # Índice: forma reducida -> palabras reales que la producen (con su frecuencia)
        self.por_reducida = {}
        for palabra, n in self.uni.items():
            self.por_reducida.setdefault(reducir(palabra), Counter())[palabra] += n
        self.vocab_reducido = sorted(self.por_reducida)
    def __contains__(self, palabra):
        return palabra in self.uni

    def logp_uni(self, palabra):
        return math.log((self.uni[palabra] + self.k) / (self.total + self.k * len(self.vocab)))

    def logp_bi(self, prev, palabra):
        return math.log((self.bi[(prev, palabra)] + self.k)
                        / (self.antes[prev] + self.k * len(self.vocab)))


    def guardar(self, ruta):
        datos = {"k": self.k, "uni": self.uni,
                 "bi": {f"{a}\t{b}": n for (a, b), n in self.bi.items()}}
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(datos, f, ensure_ascii=False)

    @classmethod
    def cargar(cls, ruta):
        with open(ruta, encoding="utf-8") as f:
            datos = json.load(f)
        lm = cls([], k=datos["k"])
        lm.uni = Counter(datos["uni"])
        lm.bi = Counter({tuple(k.split("\t")): n for k, n in datos["bi"].items()})
        lm.total = sum(lm.uni.values())
        lm.vocab = sorted(lm.uni)
        lm.antes = Counter()
        for (prev, _), n in lm.bi.items():
            lm.antes[prev] += n
        return lm
