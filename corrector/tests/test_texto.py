from corrector.texto import reduce_text, reducir


def test_reduce_text():
    assert reduce_text("canción") == "cancion"                    # sin tilde
    assert reduce_text("niño") == "niño"                          # la ñ se conserva
    assert reduce_text("pingüino") == "pinguino"                  # sin diéresis
    assert reduce_text("hola") == "ola"                           # h muda
    assert reduce_text("chico") == "chico"                        # la h de "ch" se conserva
    assert reduce_text("estoy", y_to_i=True) == "estoi"
    assert reduce_text("vaca", v_to_b=True) == "baca"


def test_reducir_usa_las_opciones_del_modelo():
    assert reducir("Hoy estoy aquí") == "oi estoi aqui"