"""
Normalización de texto y grupos de visemas

"normalize_text" la del notebook 05
el corrector tiene que limpiar las frases igual que el modelo, si no compara cosas distintas
Si el modelo cambia de normalización (por ejemplo a normalize_text_v3),
hay que cambiarla también acá
"""

import re
import unicodedata
from num2words import num2words


def normalize_text(text):
    """Igual que la celda 5 del notebook 05: minusculas, núm a palabras,
    sin puntuación (con tildes, ñ y espacios)"""
    text = text.lower()
    text = re.sub(r"\d+", lambda match: num2words(int(match.group()), lang="es"), text)
    text = "".join(char for char in text if char.isalpha() or char.isspace())
    return " ".join(text.split())

def spell_numbers(text):
    #Números a palabras, siempre separados por espacios (el original pegaba palabras)
    text = re.sub(r"(?<=\d)\.(?=\d{3}(?!\d))", "", text)                       # 10.000 -> 10000
    text = re.sub(r"(\d+)[,.](\d+)",                                              # 3,5 -> tres coma cinco
                  lambda m: f" {num2words(int(m[1]), lang='es')} coma {num2words(int(m[2]), lang='es')} ", text)
    text = text.replace("%", " por ciento ")
    text = re.sub(r"[ºª°]", " ", text)                                            # 1º, 2ª, 20°
    text = re.sub(r"\d+", lambda m: f" {num2words(int(m.group()), lang='es')} ", text)
    return text


def reduce_text(text, strip_accents=True, silent_h=True, y_to_i=False, v_to_b=False):
    #Reduce el alfabeto a lo que un modelo visual puede distinguir. Mantiene la ñ
    t = text.lower()
    if strip_accents:
        t = t.replace("ñ", "\ue000")                       # proteger la ñ
        t = unicodedata.normalize("NFD", t)
        t = "".join(c for c in t if unicodedata.category(c) != "Mn")   # quita tildes y diéresis
        t = unicodedata.normalize("NFC", t).replace("\ue000", "ñ")
    if silent_h:
        t = re.sub(r"(?<!c)h", "", t)                      # h muda; se conserva en "ch"
    if v_to_b:
        t = t.replace("v", "b")
    if y_to_i:
        t = t.replace("y", "i")
    return " ".join(t.split())


def normalize_text_v3(text, **reduce_kw):
    t = spell_numbers(text.lower())
    t = "".join(c for c in t if c.isalpha() or c.isspace())
    t = " ".join(t.split())
    return reduce_text(t, **reduce_kw)


# Opciones de reducción que usa el MODELO (tienen que coincidir con el notebook 05)
REDUCCION = dict(strip_accents=True, silent_h=True, y_to_i=True, v_to_b=False)


def reducir(texto):
    #Reduce el texto igual que el modelo
    return reduce_text(texto, **REDUCCION)



# Visemas: letras que se ven parecidas en la boca
# Es una aprox a nivel de LETRA (no de fonema): en español la
# escritura es bastante fonética, casos ambiguos (la "c" puede
# sonar /k/ o /s/), por eso una letra puede estar en más de un grupo
# mapa correcto esta en fonemas.py (VISEMA): es el que usa la cabeza de visemas
#los visemas de fonema es el que esta ahi
GRUPOS_VISEMA = {
    "bilabial":       set("pbmv"),      # labios cerrados (la v suena como b)
    "labiodental":    set("f"),         # dientes sobre el labio
    "dental_alveolar": set("tdnlsrzc"), # lengua detrás de los dientes, casi invisible
    "palatal":        set("yñ"),
    "velar":          set("kgjqcx"),    # se forman atrás, casi invisibles
    "vocal_abierta":  set("aáà"),
    "vocal_estirada": set("eéiíy"),     # labios estirados
    "vocal_redonda":  set("oóuúüw"),    # labios redondeados
}

# Pares que difieren solo en la tilde: el modelo casi no puede distinguirlos
SIN_TILDE = str.maketrans("áàéíóúü", "aaeiouu")


def mismo_visema(a, b):
    """True si las dos letras comparten algún grupo de visema."""
    return any(a in g and b in g for g in GRUPOS_VISEMA.values())


def solo_tilde(a, b):
    """True si las dos letras son la misma salvo por la tilde."""
    return a != b and a.translate(SIN_TILDE) == b.translate(SIN_TILDE)
