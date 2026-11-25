from corrector.fonemas import a_fonemas, pronunciaciones, VISEMA

def test_letras_simples():
    assert a_fonemas("mesa") == ["m", "e", "s", "a"]
    assert a_fonemas("vaca") == ["b", "a", "k", "a"]
    assert a_fonemas("zapato") == ["s", "a", "p", "a", "t", "o"]
    assert a_fonemas("niño") == ["n", "i", "ɲ", "o"]
    assert a_fonemas("frac") == ["f", "ɾ", "a", "k"]

def test_g():
    assert a_fonemas("gato") == ["g", "a", "t", "o"]
    assert a_fonemas("gente") == ["x", "e", "n", "t", "e"]
    assert a_fonemas("girafa") == ["x", "i", "ɾ", "a", "f", "a"]
    assert a_fonemas("grande") == ["g", "ɾ", "a", "n", "d", "e"]

def test_gu():
    assert a_fonemas("guitarra") == ["g", "i", "t", "a", "r", "a"]
    assert a_fonemas("guerra") == ["g", "e", "r", "a"]
    assert a_fonemas("pingüino") == ["p", "i", "n", "g", "w", "i", "n", "o"]
    assert a_fonemas("agua") == ["a", "g", "u", "a"]

def test_qu():
    assert a_fonemas("queso") == ["k", "e", "s", "o"]
    assert a_fonemas("quien") == ["k", "i", "e", "n"]
    assert a_fonemas("tranquilo") == ["t", "ɾ", "a", "n", "k", "i", "l", "o"]

def test_ll_y():
    assert a_fonemas("llave") == ["ʝ", "a", "b", "e"]
    assert a_fonemas("yo") == ["ʝ", "o"]
    assert a_fonemas("ayer") == ["a", "ʝ", "e", "ɾ"]
    assert a_fonemas("calle") == ["k", "a", "ʝ", "e"]
    assert a_fonemas("hoy") == ["o", "i"]      # este va a fallar hasta que hagas la h
    assert a_fonemas("muy") == ["m", "u", "i"]
    assert a_fonemas("y") == ["i"]

def test_combinaciones():
    assert a_fonemas("chico") == ["tʃ", "i", "k", "o"]
    assert a_fonemas("hola") == ["o", "l", "a"]
    assert a_fonemas("hoy") == ["o", "i"]
    assert a_fonemas("perro") == ["p", "e", "r", "o"]
    assert a_fonemas("taxi") == ["t", "a", "k", "s", "i"]
    assert a_fonemas("accidente") == ["a", "k", "s", "i", "d", "e", "n", "t", "e"]
    assert a_fonemas("canción") == ["k", "a", "n", "s", "i", "o", "n"]

def test_r():
    assert a_fonemas("pero") == ["p", "e", "ɾ", "o"]
    assert a_fonemas("perro") == ["p", "e", "r", "o"]
    assert a_fonemas("rosa") == ["r", "o", "s", "a"]
    assert a_fonemas("honra") == ["o", "n", "r", "a"]
    assert a_fonemas("amor") == ["a", "m", "o", "ɾ"]
    assert a_fonemas("guitarra") == ["g", "i", "t", "a", "r", "a"]


def test_variantes():
    assert a_fonemas("calle") == ["k", "a", "ʝ", "e"]                       # general, por defecto
    assert a_fonemas("calle", "rioplatense") == ["k", "a", "ʃ", "e"]
    assert a_fonemas("mosca", "rioplatense") == ["m", "o", "h", "k", "a"]
    assert a_fonemas("hoy", "rioplatense") == ["o", "i"]                    # y vocal: no cambia
    assert a_fonemas("casa", "rioplatense") == ["k", "a", "s", "a"]         # s antes de vocal: no cambia
    assert a_fonemas("más", "rioplatense") == ["m", "a", "s"]               # s final: no cambia


def test_pronunciaciones():
    assert pronunciaciones("casa") == {("k", "a", "s", "a")}
    assert pronunciaciones("calle") == {("k", "a", "ʝ", "e"), ("k", "a", "ʃ", "e")}