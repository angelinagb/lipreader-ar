"""
Distancia de Levenshtein con costos que entienden de visemas

En la distancia común, cambiar "p" por "b" cuesta lo mismo que cambiar
"p" por "s". Pero un lector de labios confunde más p/b/m (se ven
iguales) que p/s. Con estos costos, una corrección que solo cambia letras
del mismo visema queda "más cerca" y gana
"""
from functools import lru_cache

from .fonemas import VISEMA, a_fonemas
from .texto import mismo_visema, solo_tilde
import math
from collections import Counter
from .confusion import VACIO


class Costos:
    #Costos de cada operación. Los valores son un punto de partida: se ajustan con val

    def __init__(self, mismo_visema=0.4, tilde=0.1, h_muda=0.3,
                 sustitucion=1.0, insercion=1.0, borrado=1.0):
        self.mismo_visema = mismo_visema   # p<->b, o<->u...
        self.tilde = tilde                 # a<->á
        self.h_muda = h_muda               # agregar o sacar una h (no se ve ni se oye)
        self.sustitucion = sustitucion
        self.insercion = insercion
        self.borrado = borrado

    def sub(self, a, b):
        if a == b:
            return 0.0
        if solo_tilde(a, b):
            return self.tilde
        if mismo_visema(a, b):
            return self.mismo_visema
        return self.sustitucion

    def ins(self, c):
        return self.h_muda if c == "h" else self.insercion

    def dele(self, c):
        return self.h_muda if c == "h" else self.borrado


COSTOS_PLANOS = Costos(mismo_visema=1.0, tilde=1.0, h_muda=1.0)


def distancia_ponderada(origen, destino, costos):
    #Minimo costo para transformar "origen" (lo que predijo el modelo) en "destino"
    prev = [0.0]
    for c in destino:
        prev.append(prev[-1] + costos.ins(c))
    for a in origen:
        cur = [prev[0] + costos.dele(a)]
        for j, b in enumerate(destino, start=1):
            cur.append(min(
                prev[j] + costos.dele(a),          # borrar a
                cur[j - 1] + costos.ins(b),        # insertar b
                prev[j - 1] + costos.sub(a, b),    # cambiar a por b
            ))
        prev = cur
    return prev[-1]

@lru_cache(maxsize=None)
def _fonemas(palabra):
    #Guarda los fonemas de cada palabra para no recalcularlos 
    return tuple(a_fonemas(palabra))


def distancia_fonemas(origen, destino, mismo_visema=0.4):
    #Levenshtein sobre fonemas: igual fonema = 0, mismo visema = barato, distinto = 1
    a, b = _fonemas(origen), _fonemas(destino)
    prev = [float(j) for j in range(len(b) + 1)]
    for i, fa in enumerate(a, start=1):
        cur = [float(i)]
        for j, fb in enumerate(b, start=1):
            if fa == fb:
                sub = 0.0
            elif VISEMA[fa] == VISEMA[fb]:
                sub = mismo_visema
            else:
                sub = 1.0
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + sub))
        prev = cur
    return prev[-1]

def costos_aprendidos(conteo, k=0.1, min_casos=20):
    #conteo: Counter {(real, pred): n}. Devuelve {(real, pred): costo}
    simbolos = sorted({r for r, _ in conteo} | {p for _, p in conteo})
    V = len(simbolos)
    total = Counter()
    for (r, p), n in conteo.items():
        total[r] += n

    costos = {}
    for r in simbolos:
        if total[r] < min_casos:
            continue            
        for p in simbolos:
            if r == VACIO and p == VACIO:
                continue
            prob = (conteo.get((r, p), 0) + k) / (total[r] + k * V)
            costos[(r, p)] = -math.log(prob)
    return costos

def distancia_aprendida(pred, palabra, costos):
    def c(real, p):
        if (real, p) in costos:
            return costos[(real, p)]
        return 0.0 if real == p else 1.0

    n, m = len(palabra), len(pred)
    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = D[i-1][0] + c(palabra[i-1], VACIO)  # # el modelo se comió palabra[i-1]
    for j in range(1, m + 1):
        D[0][j] = D[0][j-1] + c(VACIO, pred[j-1])      # el modelo inventó pred[j-1]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(
                D[i-1][j]   + c(palabra[i-1], VACIO),     # se comió
                D[i][j-1]   + c(VACIO, pred[j-1]),     # inventó
                D[i-1][j-1] + c(palabra[i-1], pred[j-1]),     # vio palabra[i-1] y escribió pred[j-1]
            )
    return D[n][m]
