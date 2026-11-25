from corrector.confusion import alinear, VACIO, contar_confusiones


def test_alinear():
    assert alinear("pata", "bata") == [("p", "b"), ("a", "a"), ("t", "t"), ("a", "a")]
    assert alinear("casa", "cas") == [("c", "c"), ("a", "a"), ("s", "s"), ("a", VACIO)]
    assert alinear("sol", "sool") == [("s", "s"), (VACIO, "o"), ("o", "o"), ("l", "l")]
    assert alinear("", "ab") == [(VACIO, "a"), (VACIO, "b")]
    assert alinear("ab", "") == [("a", VACIO), ("b", VACIO)]

def test_contar():
    c = contar_confusiones(["pata", "pala"], ["bata", "bala"])
    assert c[("p", "b")] == 2
    assert c[("a", "a")] == 4