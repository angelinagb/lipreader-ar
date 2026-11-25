#Matriz de confusión: qué letras confunde el modelo

from collections import Counter
import pandas as pd

VACIO = "∅"   


def contar_pares(reales, preds):
    #Alinea cada frase real con su predicción y cuenta los pares (real, pred)
    conteo = Counter()
    for r, p in zip(reales, preds):
        for a, b in alinear(list(r), list(p)):
            conteo[(a, b)] += 1
    return conteo
def tabla_levenshtein(real, pred):
    #Tabla completa de Levenshtein: D[i][j] = costo de real[:i] -> pred[:j]
    n, m = len(real), len(pred)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            costo = 0 if real[i - 1] == pred[j - 1] else 1
            D[i][j] = min(D[i - 1][j] + 1,          # borrado
                          D[i][j - 1] + 1,          # inserción
                          D[i - 1][j - 1] + costo)  # acierto o sustitución
    return D

def alinear(real, pred):
    #Devuelve la lista de pares (letra real, letra predicha), en orden
    D = tabla_levenshtein(real, pred)
    i, j = len(real), len(pred)
    pares = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (0 if real[i - 1] == pred[j - 1] else 1):
            pares.append((real[i - 1], pred[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            pares.append((real[i - 1], VACIO))
            i -= 1
        else:
            pares.append((VACIO, pred[j - 1]))
            j -= 1
    return pares[::-1]   

def contar_confusiones(referencias, predicciones):
    #Cuenta los pares (real, predicho) de todas las frases
    conteo = Counter()
    for real, pred in zip(referencias, predicciones):
        conteo.update(alinear(real, pred))
    return conteo


def matriz_confusion(conteo, orden=None):
    #DataFrame con filas = real, columnas = predicho donde cada fila suma 1
    filas = sorted({r for r, _ in conteo} | {p for _, p in conteo})
    orden = orden or filas
    m = pd.DataFrame(0, index=orden, columns=orden, dtype=float)
    for (r, p), n in conteo.items():
        if r in m.index and p in m.columns:
            m.loc[r, p] += n
    return m.div(m.sum(axis=1).replace(0, 1), axis=0)