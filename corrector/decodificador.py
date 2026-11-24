"""
Decodificador de frase entera.

En vez de corregir palabra por palabra, busca la secuencia de palabras del
diccionario que mejor explica TODA la secuencia de letras predicha (sin espacios):
costo = sum( distancia_aprendida) + lam * sum( −log P(w | anterior) ) + beta * (cant de palabras)
como el bigrama solo mira la última palabra,
en cada posición basta guardar la mejor hipótesis por cada última palabra.
"""
from functools import lru_cache

from .confusion import VACIO
from .lenguaje import INICIO





def decodificar(pred, lm, candidatas, lam=1.0, beta=1.0, B=10, LMAX=4, silenciosas=(), bono_sil=0.0):    
    """
    candidatas(f)  lista de (palabra, distancia) para el pedazo f.
    silenciosas    lista de (palabra, costo_de_borrarla_entera): palabras que el
                     modelo suele comerse por completo. Se pueden insertar sin
                     consumir letras (como mucho una seguida).
    """
    s = pred.replace(" ", "")
    if not s:
        return ""
    mejores = [dict() for _ in range(len(s) + 1)]
    mejores[0][INICIO] = (0.0, ())

    def agregar(destino, w, nuevo, palabras):
        if w not in destino or nuevo < destino[w][0]:
            destino[w] = (nuevo, palabras + (w,))

    for j in range(len(s)):
        vivas = sorted(mejores[j].items(), key=lambda kv: kv[1][0])[:B]

        # insertar una palabra silenciosa antes de seguir
        con_silenciosa = {}
        for anterior, (costo, palabras) in vivas:
            for w, d in silenciosas:
                nuevo = costo + d + lam * -lm.logp_bi(anterior, w) + beta + bono_sil
                agregar(con_silenciosa, w, nuevo, palabras)
        extra = sorted(con_silenciosa.items(), key=lambda kv: kv[1][0])[:B]

        # Consumir un pedazo de s, partiendo de las vivas o de las que tienen silenciosa
        for anterior, (costo, palabras) in vivas + extra:
            for L in range(1, min(LMAX, len(s) - j) + 1):
                f = s[j:j + L]
                for w, d in candidatas(f):
                    nuevo = costo + d + lam * -lm.logp_bi(anterior, w) + beta
                    agregar(mejores[j + L], w, nuevo, palabras)

    if not mejores[len(s)]:
        return pred
    return " ".join(min(mejores[len(s)].values())[1])