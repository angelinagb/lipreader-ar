#Conversion de palabras en español a fonemas (seseo y yeísmo)

# Letras que se convierten siempre igual, sin mirar la siguiente
SIMPLES = {
    "a": "a", "e": "e", "i": "i", "o": "o", "u": "u",
    "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ü": "u",
    "b": "b", "v": "b",
    "d": "d", "f": "f", "k": "k", "l": "l", "m": "m",
    "n": "n", "ñ": "ɲ", "p": "p", "s": "s", "t": "t",
    "z": "s", "j": "x", "w": "w","à": "a"
}

VARIANTES = ("general", "rioplatense")
VOCALES = "aeiouáéíóú"

INICIO = "^"   # marca de comienzo de palabra (al lado de FIN = "#")

FIN = "#"   # marca de fin de palabra (EOS)


def a_fonemas(palabra, variante="general"):
    if variante not in VARIANTES:
        raise ValueError(f"variante debe ser una de {VARIANTES}, no {variante!r}")
    rio = variante == "rioplatense"
    fonemas = []
    i = 0
    while i < len(palabra):
        letra = palabra[i]
        siguiente = palabra[i + 1] if i + 1 < len(palabra) else FIN
        anterior = palabra[i - 1] if i > 0 else INICIO
        if letra == "c":
            if siguiente in "eéií":
                fonemas.append("s")      # cena, cine
            elif siguiente == "h":
                fonemas.append("tʃ")     # chocolate, chicle
                i += 1
            else:
                fonemas.append("k")      # casa, claro, y la c final
            i += 1
            continue
        if letra == "g":
            if siguiente in "eéií":
                fonemas.append("x")      # gente, girasol
            else:
                if siguiente == "u" and i + 2 < len(palabra) and palabra[i + 2] in "eéií":
                    fonemas.append("g")  # guerra, guitarra
                    i += 1                # saltamos la u muda
                elif siguiente == "ü" and i + 2 < len(palabra) and palabra[i + 2] in "eéií":
                    fonemas.append("g")  # pingüino
                    fonemas.append("w")      # pingüino
                    i += 1                
                else:
                    fonemas.append("g")      # gato, gol, y la g final
            i += 1
            continue
        if letra=="q":
            fonemas.append("k")          # queso, quimera
            i += 2  #salto la u muda
            continue
        if letra == "h":
            i += 1  # la h es muda
            continue
      
        if letra=="y" :
            if siguiente in VOCALES:
                fonemas.append("ʃ" if rio else "ʝ")   # yo -> "sho" en rioplatense
            else:
                fonemas.append("i")                    # hoy, y: igual en las dos
            i += 1
            continue
        if letra=="l" and siguiente=="l":
            fonemas.append("ʃ" if rio else "ʝ")       # calle -> "cashe" en rioplatense
            i += 2
            continue
        if letra == "r":
            if siguiente == "r":
                fonemas.append("r")      # perro, carro (fuerte)
                i += 2                   # rr = un solo sonido
                continue
            if anterior == INICIO or anterior in "nls":
                fonemas.append("r")      # rosa, honra, alrededor (fuerte)
            else:
                fonemas.append("ɾ")      # pero, cara, amor (suave)
            i += 1
            continue
        if letra=="x":
            fonemas.extend(["k", "s"])         # taxi, examen
            i += 1
            continue
        if letra == "s" and rio and siguiente not in VOCALES + FIN:
            fonemas.append("h")      # mosca -> mohca, esto -> ehto
            i += 1
            continue
        if letra in SIMPLES:
            fonemas.append(SIMPLES[letra])
        i += 1
    return fonemas

VISEMA = {
    # Bilabiales
    "p": "bilabial",
    "b": "bilabial",
    "m": "bilabial",
    "w": "bilabial_velar",  # diptongos con u (ej "huevo", "agua")

    # Labiodental
    "f": "labiodental",

    # Dentales (en español t y d son dentales)
    "t": "dental",
    "d": "dental",

    # Alveolares
    "n": "alveolar",
    "s": "alveolar",
    "l": "alveolar",
    "ɾ": "alveolar",  # r simple (ej "caro")
    "r": "alveolar",  # rr múltiple (ej "carro")

    # Postalveolares
    "tʃ": "postalveolar",  # ej "chico"
    "ʃ": "postalveolar",   # solo rioplatense: ll / y rehiladas (ej "calle" -> "cashe")

    # Palatales
    "ʝ": "palatal",  # ll / y con yeísmo general (ej. "calle", "yo")
    "ɲ": "palatal",  # ej "ñandu"

    # Velares
    "k": "velar",
    "g": "velar",
    "x": "velar",  # ej "jamon", "gente"

    # Glotal
    "h": "glotal",  # solo rioplatense: s aspirada (ej "mosca" -> "mohca" parecido a mojca)

    # Vocales
    "a": "vocal_a",
    "e": "vocal_e",
    "i": "vocal_i",
    "o": "vocal_o",
    "u": "vocal_u",
}


def a_visemas(palabra, variante="general"):
    #Convierte una palabra en su secuencia de visemas
    return [VISEMA[f] for f in a_fonemas(palabra, variante)]


def pronunciaciones(palabra):
    #Todas las pronunciaciones distintas de la palabra (general y rioplatense)
    return {tuple(a_fonemas(palabra, v)) for v in VARIANTES}


PAUSA = "pausa"
VISEMAS = sorted(set(VISEMA.values())) + [PAUSA]
VISEMA_A_IDX = {v: i for i, v in enumerate(VISEMAS, start=1)}   # 0 = blank

def fraseAVisema(frase):
        #Convierte una frase en una lista de visemas, con pausas entre palabras
        visemas = []
        for palabra in frase.split():
            visemas.extend(a_visemas(palabra))
            visemas.append(PAUSA)
        return visemas[:-1]  

def codificarVisemas(visemas):
   #Convierte una lista de visemas en una lista de indices, 0 = blank
    return [VISEMA_A_IDX[v] for v in visemas]

"""
bilabial: los labios superior e inferior se juntan

labiodental: el labio inferior toca o se acerca a los dientes superiores (lo contrario, que el labio superior toque los dientes inferiores, es posible pero poco común)

dental: la punta de la lengua toca o se acerca a los dientes, ya sea entre los dientes superiores e inferiores o detrás de los dientes superiores

alveolar: la punta de la lengua toca o se acerca a la cresta de la encía que se encuentra justo detrás de los dientes superiores (la cresta alveolar)

postalveolar: La lengua forma un paso estrecho justo detrás de la cresta alveolar

palatal: la parte central de la lengua se eleva hacia el paladar duro, la parte ósea del techo de la boca

velar: la parte posterior de la lengua toca o se acerca a la velo (paladar blando)

glotal: el sonido se produce en el glottis, la abertura entre las cuerdas vocales: al cerrarla brevemente se produce el sonido ʔ. Para la h, las cuerdas vocales se mantienen separadas para permitir que el aire fluya libremente

labio-velar: dos constricciones a la vez: los labios se redondean mientras que la parte posterior de la lengua se eleva hacia el velo del paladar, como en la «w». Ninguna de las dos se cierra por completo, lo que la convierte en una aproximante 
"""