"""Tests chicos que no necesitan el parquet. Correr con:  python -m pytest corrector/tests"""
from corrector.distancia import distancia_fonemas
from corrector import Corrector, Costos, ModeloLenguaje, normalize_text
from corrector.distancia import COSTOS_PLANOS, distancia_ponderada
from corrector.metricas import cer_wer, edit_distance

FRASES = [
    "la ficción es una herramienta poderosa",
    "una herramienta muy poderosa",
    "la ola del mar",
    "hola cómo estás",
    "hola a todos",
    "la ficción nos ayuda",
]


def test_normalizacion_igual_al_notebook():
    assert normalize_text("¡Hola, Mundo! Tengo 20 años.") == "hola mundo tengo veinte años"


def test_edit_distance():
    assert edit_distance("hola", "ola") == 1
    assert edit_distance("hola mundo".split(), "ola mundo".split()) == 1


def test_cer_wer_perfecto():
    assert cer_wer(["hola mundo"], ["hola mundo"]) == (0.0, 0.0)


def test_visema_mas_barato():
    con = Costos()
    assert distancia_ponderada("pala", "bala", con) < distancia_ponderada("pala", "sala", con)
    assert distancia_ponderada("pala", "bala", COSTOS_PLANOS) == 1.0


def test_tilde_y_h_baratas():
    con = Costos()
    assert distancia_ponderada("ficcion", "ficción", con) < 0.5
    assert distancia_ponderada("erramienta", "herramienta", con) < 0.5


def test_corrige_palabras_desconocidas():
    lm = ModeloLenguaje(FRASES)
    corr = Corrector(lm)
    assert corr.corregir("la fiscion es una erramienta poderoza") == \
        "la ficción es una herramienta poderosa"


def test_deja_palabras_conocidas_por_defecto():
    lm = ModeloLenguaje(FRASES)
    assert Corrector(lm).corregir("ola a todos") == "ola a todos"


def test_contexto_corrige_conocidas():
    lm = ModeloLenguaje(FRASES)
    corr = Corrector(lm, peso_contexto=1.0, corregir_conocidas=True)
    assert corr.corregir("ola a todos") == "hola a todos"


def test_guardar_y_cargar(tmp_path):
    lm = ModeloLenguaje(FRASES)
    ruta = tmp_path / "lm.json"
    lm.guardar(ruta)
    lm2 = ModeloLenguaje.cargar(ruta)
    assert lm2.vocab == lm.vocab and lm2.logp_bi("la", "ola") == lm.logp_bi("la", "ola")

def test_distancia_fonemas():
    assert distancia_fonemas("sena", "cena") == 0      # suenan igual
    assert distancia_fonemas("ola", "hola") == 0       # la h no suena
    assert distancia_fonemas("baca", "vaca") == 0      # b y v suenan igual
    assert distancia_fonemas("pata", "bata") == 0.4    # mismo visema (bilabial)
    assert distancia_fonemas("pata", "gata") == 1.0    # distinto visema

def test_corrector_con_fonemas():
    lm = ModeloLenguaje(FRASES)
    corr = Corrector(lm, comparar="fonemas")
    assert corr.corregir("la fiscion es una erramienta poderoza") == \
        "la ficción es una herramienta poderosa"

def test_indice_reducido():
    lm = ModeloLenguaje(["hola a todos", "la ola del mar", "hoy estoy aquí"])
    assert lm.por_reducida["ola"] == {"hola": 1, "ola": 1}
    assert lm.por_reducida["estoi"] == {"estoy": 1}
    assert "aqui" in lm.vocab_reducido

def test_modo_reducido():
    lm = ModeloLenguaje(["hola a todos", "la ola del mar", "hoy estoy aquí con mi hermano"])
    corr = Corrector(lm, reducido=True, peso_contexto=1.0)
    assert corr.corregir("oi estoi aqui con mi ermano") == "hoy estoy aquí con mi hermano"
    assert corr.corregir("ola a todos") == "hola a todos"
    assert corr.corregir("la ola del mar") == "la ola del mar"