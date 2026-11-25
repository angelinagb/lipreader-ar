"""
La palabra que se muestra se elige según
qué tan parecida es la palabra candidata a la predicha,
qué tan común/frecuente es y qué tan bien sigue a la antrior 
(prob condicional)
Predicciones muy rotas quedan igual, palabras ue no están en train no se pueden recurepara
y no une o separa palabras, solo corrige errores de ortografía
"""
import math
from functools import lru_cache
from .texto import reducir
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
from .distancia import Costos, distancia_aprendida, distancia_fonemas, distancia_ponderada
from .lenguaje import INICIO
class Corrector:
    def __init__(self, lm, costos=None, max_dist_rel=0.4, n_candidatos=150,
                 peso_freq=0.05, peso_contexto=0.0, corregir_conocidas=False,
                 comparar="letras", alfa=0.5, costo_desconocida=None, reducido=False, costos_conf=None):
        """
        lm                  ModeloLenguaje armado con las frases de train
        costos              Costos de la distancia (None = con visemas, valores por defecto)
        max_dist_rel        corrige solo si la candidata está a lo sumo a este % de letras
                            cambiadas (0.4 = una palabra de 5 letras admite 2 cambios)
        n_candidatos        cuántas candidatas parecidas se miran por palabra
        peso_freq           cuánto pesa que la palabra sea común
        peso_contexto       cuánto pesa el bigrama (0 = palabra por palabra)
        corregir_conocidas  si True, también puede cambiar palabras que SÍ existen
                            (necesario para "ola" -> "hola"); solo tiene sentido con contexto
        comparar            "letras" (ortografía) o "fonemas" (cómo suena y se ve)
            alfa                con comparar="combinada": peso de las letras (0 = solo fonemas, 1 = solo letras)
        costo_desconocida   si no es None, una palabra fuera del diccionario puede quedar igual,
                            pagando este puntaje (evita "corregir" palabras que estaban bien)    
               reducido            si True, la entrada viene reducida como la del modelo (sin tildes ni h,
                            y igual a i) y el corrector devuelve la ortografía real
         """
        if comparar not in ("letras", "fonemas","combinada", "aprendida"):
            raise ValueError("comparar debe ser 'letras' o 'fonemas' o 'combinada' o 'aprendida'")
        if comparar == "aprendida" and costos_conf is None:
            raise ValueError("comparar='aprendida' necesita costos_conf")
        self.lm = lm
        self.costos = costos or Costos()
        self.max_dist_rel = max_dist_rel
        self.n_candidatos = n_candidatos
        self.peso_freq = peso_freq
        self.peso_contexto = peso_contexto
        self.corregir_conocidas = corregir_conocidas
        self.comparar = comparar
        self.alfa = alfa
        self.costo_desconocida = costo_desconocida
        self.costos_conf = costos_conf
        self._candidatas = lru_cache(maxsize=200_000)(self._buscar_candidatas)
        self.reducido = reducido
    def _buscar_candidatas(self, palabra):
        #Palabras del diccionario a distancia (común) chica, con su distancia ponderada
        if self.comparar == "aprendida":
            tope = len(palabra)
        else:
            tope = max(1, math.ceil(len(palabra) * self.max_dist_rel))
        vocab = self.lm.vocab_reducido if self.reducido else self.lm.vocab
        cercanas = process.extract(palabra, vocab, scorer=Levenshtein.distance,
                                   limit=self.n_candidatos, score_cutoff=tope)
        resultado = []
        for c, _, _ in cercanas:
            if self.comparar == "letras":
                d = distancia_ponderada(palabra, c, self.costos)
            elif self.comparar == "fonemas":
                d = distancia_fonemas(palabra, c, self.costos.mismo_visema)
            elif self.comparar == "aprendida":
                d = distancia_aprendida(palabra, c, self.costos_conf)
            else:
                # combinada
                d = (self.alfa * distancia_ponderada(palabra, c, self.costos)
                     + (1 - self.alfa) * distancia_fonemas(palabra, c, self.costos.mismo_visema))
            resultado.append((c, d))
        return tuple(resultado)
    def _puntaje(self, distancia, candidata, anterior):
        p = distancia + self.peso_freq * -self.lm.logp_uni(candidata)
        if self.peso_contexto:
            p += self.peso_contexto * -self.lm.logp_bi(anterior, candidata)
        return p

    def _corregir_reducida(self, palabra, anterior):
        #Modo reducido: la palabra viene sin tildes, sin h, etc. Devuelve la palabra REAL
        conocida = palabra in self.lm.por_reducida
        if conocida and not self.corregir_conocidas:
            candidatas = [(palabra, 0.0)]          # la forma reducida existe: solo falta elegir la real
        else:
            candidatas = list(self._candidatas(palabra))
            if conocida:
                candidatas.append((palabra, 0.0))
        if not candidatas:
            return palabra
        # Cada forma reducida candidata se expande a sus palabras reales ("ola" -> "hola", "ola")
        reales = [(real, d) for red, d in candidatas for real in self.lm.por_reducida[red]]
        return min(reales, key=lambda rd: self._puntaje(rd[1], rd[0], anterior))[0]

    def corregir_palabra(self, palabra, anterior=INICIO):
        if self.reducido:
            return self._corregir_reducida(palabra, anterior)
        conocida = palabra in self.lm
        if conocida and not self.corregir_conocidas:
            return palabra

        candidatas = list(self._candidatas(palabra))
        if conocida:
            candidatas.append((palabra, 0.0))
        if not candidatas:
            return palabra   # nada parecido en el diccionario: se deja igual

        mejor = min(candidatas, key=lambda cd: self._puntaje(cd[1], cd[0], anterior))
        # Opción "dejarla como está" para palabras fuera del diccionario
        if (not conocida and self.costo_desconocida is not None
                and self.costo_desconocida < mejor[1]):
            return palabra
        return mejor[0]

    def corregir(self, frase):
        if self.reducido:
            frase = reducir(frase)
        corregidas, anterior = [], INICIO
        for palabra in frase.split():
            nueva = self.corregir_palabra(palabra, anterior)
            corregidas.append(nueva)
            anterior = nueva
        return " ".join(corregidas)
