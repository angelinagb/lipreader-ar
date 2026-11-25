"""
Errores simulados: "rompe" frases correctas para probar el corrector
sin depender de las predicciones del modelo

Dos modos:
- "azar":   cambia letras por cualquier otra
- "visema": cuando cambia una letra, el 70% de las veces elige una que
            se ve igual en la boca (p->b, o->u). Se parece más a cómo se
            equivoca un lector de labios de verdad
No toca los espacios, así que la cantidad de palabras se mantiene
"""

import random

from .texto import GRUPOS_VISEMA

LETRAS = "abcdefghijklmnopqrstuvwxyzñáéíóú"


def _parecida(c, rng):
    grupos = [g for g in GRUPOS_VISEMA.values() if c in g and len(g) > 1]
    if not grupos:
        return rng.choice(LETRAS)
    opciones = sorted(rng.choice(grupos) - {c})
    return rng.choice(opciones)


def romper(frase, tasa, modo="visema", rng=None):
    #Aplica errores a aprox "tasa" de las letras (0.3 = 30%)
    rng = rng or random.Random(0)
    salida = []
    for c in frase:
        if c == " " or rng.random() >= tasa:
            salida.append(c)
            continue
        op = rng.random()
        if op < 0.6:                                   # sustitución
            if modo == "visema" and rng.random() < 0.7:
                salida.append(_parecida(c, rng))
            else:
                salida.append(rng.choice(LETRAS))
        elif op < 0.85:                                # borrado
            pass
        else:                                          # inserción
            salida.append(c)
            salida.append(rng.choice(LETRAS))
    return " ".join("".join(salida).split())
