#CER y WER con distancia de Levenshtein


def edit_distance(ref, hyp):
    #Levenshtein: sirve para strings (CER) y para listas de palabras (WER)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i]
        for j, h in enumerate(hyp, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def cer_wer(referencias, hipotesis):
    #CER y WER globales: errores totales / largo total
    c_err = c_tot = w_err = w_tot = 0
    for ref, hyp in zip(referencias, hipotesis):
        c_err += edit_distance(ref, hyp)
        c_tot += len(ref)
        w_err += edit_distance(ref.split(), hyp.split())
        w_tot += len(ref.split())
    return c_err / max(c_tot, 1), w_err / max(w_tot, 1)
