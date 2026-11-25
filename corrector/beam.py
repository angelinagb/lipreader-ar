"""
Beam search de CTC con modelo de lenguaje de palabras (bigramas)

El argmax elige la mejor letra en cada frame por separado y tira el resto
Beam search recorre los frames manteniendo las B mejores frases parciales
(prefijos), usando TODAS las probabilidades del modelo. Cada vez que un
prefijo cierra una palabra (agrega un espacio), se le suma alfa * log P(palabra | anterior) + beta
para empujar hacia frases que tengan sentido en español.


"""
import math

import numpy as np

from .lenguaje import INICIO

NEG = -1e30   # "log de 0"


def lse(a, b):
    #log(e^a + e^b) sin desborde
    if a < b:
        a, b = b, a
    if b <= NEG:
        return a
    return a + math.log1p(math.exp(b - a))


def _puntaje_palabra(prefijo, lm, alfa, beta):
    #LM de la ultima palabra completa del prefijo (el que termina en espacio)
    palabras = prefijo.split()
    if not palabras or lm is None:
        return 0.0
    anterior = palabras[-2] if len(palabras) > 1 else INICIO
    return alfa * lm.logp_bi(anterior, palabras[-1]) + beta


def prefijos_de(vocab):
    #Todos los comienzos posibles de palabras del diccionario ("h", "ho", "hol", "hola",etc)
    return {w[:i] for w in vocab for i in range(1, len(w) + 1)}


def beam_ctc(lp, idx_to_char, blank, lm=None, alfa=0.5, beta=0.0, B=16, top_c=8, prefijos=None):
    """
    lp           matriz (T, V) de log-probabilidades del modelo para UN solo clip
    idx_to_char  {indice: letra}
    blank        indice del blank
    lm           ModeloLenguaje (None = beam search sin modelo de lenguaje)
    alfa, beta   peso del bigrama y bono por palabra
    B            cuantos prefijos se mantienen por frame
    top_c        cuantas letras se prueban por frame (las más probables)
    prefijos     si se pasa (ver prefijos_de), solo se permiten palabras del diccionario:
                 un prefijo en una palabra en curso que no es comienzo de ninguna palabra se descarta,
                 y solo se puede cerrar una palabra (espacio) si es una palabra real
    """
    # prefijo : log P(terminar en blank), log P(terminar en letra)
    beams = {"": [0.0, NEG]}
    lm_pref = {"": 0.0}                        # puntaje LM acumulado de cada prefijo

    for t in range(lp.shape[0]):
        fila = lp[t]
        letras = np.argsort(fila)[-top_c:]
        nuevos = {}

        def sumar(pref, pos, valor):
            par = nuevos.setdefault(pref, [NEG, NEG])
            par[pos] = lse(par[pos], valor)

        for pref, (pb, pnb) in beams.items():
            total = lse(pb, pnb)
            sumar(pref, 0, total + fila[blank])                    # blank: el prefijo no cambia
            ultima = pref[-1] if pref else None
            for ci in letras:
                if ci == blank:
                    continue
                c, p = idx_to_char[ci], float(fila[ci])
                if c == " " and (not pref or ultima == " "):
                    continue                                       # sin espacios al inicio ni dobles
                if prefijos is not None:
                    actual = pref.split(" ")[-1]
                    if c == " " and actual not in lm:
                        continue                                   # cerrar solo palabras reales
                    if c != " " and c != ultima and actual + c not in prefijos:
                        continue                                   # no seguir algo que no lleva a ninguna palabra
                if c == ultima:
                    sumar(pref, 1, pnb + p)                        # misma letra seguida: se colapsa
                    if prefijos is not None and c != " " and pref.split(" ")[-1] + c not in prefijos:
                        continue                                   #    ("rr" solo si existe la palabra)
                    ext, desde = pref + c, pb                      #    letra doble solo si hubo blank en medio
                else:
                    ext, desde = pref + c, total                   # letra nueva
                sumar(ext, 1, desde + p)
                if ext not in lm_pref:
                    extra = _puntaje_palabra(ext, lm, alfa, beta) if c == " " else 0.0
                    lm_pref[ext] = lm_pref[pref] + extra

        # quedarse con los B mejores (probabilidad del modelo + modelo de lenguaje)
        beams = dict(sorted(nuevos.items(),
                            key=lambda kv: lse(*kv[1]) + lm_pref[kv[0]],
                            reverse=True)[:B])

    def final(kv):
        pref, par = kv
        cierre = 0.0 if pref.endswith(" ") else _puntaje_palabra(pref + " ", lm, alfa, beta)
        if prefijos is not None and pref.strip() and pref.split()[-1] not in lm:
            cierre -= 100.0                                        # terminó a mitad de palabra
        return lse(*par) + lm_pref[pref] + cierre

    return max(beams.items(), key=final)[0].strip()


def greedy_ctc(lp, idx_to_char, blank):
    #argmax + colapso: lo mismo que hace el notebook 
    salida, anterior = [], None
    for i in lp.argmax(axis=1):
        if i != anterior and i != blank:
            salida.append(idx_to_char[i])
        anterior = i
    return " ".join("".join(salida).split())