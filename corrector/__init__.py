"""Corrector de transcripciones para lipreader-ar (diccionario + Levenshtein + bigramas)."""

from .corrector import Corrector
from .distancia import COSTOS_PLANOS, Costos
from .lenguaje import ModeloLenguaje
from .texto import normalize_text