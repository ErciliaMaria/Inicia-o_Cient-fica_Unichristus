import csv
from pathlib import Path

import numpy as np
import pandas as pd

ARQUIVO_DADOS = Path(__file__).resolve().parent / 'heart_disease_cleveland.csv'
COLUNA = 'age'
LIMITES = (0.0, 120.0)
EPSILONS = (0.1, float(np.log(2)), float(np.log(4)), 1.0)
N_REPETICOES = 200
SEMENTE = 20260909


def carregar_idades():
    if not ARQUIVO_DADOS.exists():
        raise FileNotFoundError(f"O arquivo '{ARQUIVO_DADOS}' nao foi encontrado.")

    dados = pd.read_csv(ARQUIVO_DADOS)
    if COLUNA not in dados.columns:
        raise ValueError(f"A coluna '{COLUNA}' nao existe no CSV.")

    idades = dados[COLUNA].dropna().to_numpy(dtype=float)
    if len(idades) == 0:
        raise ValueError(f"A coluna '{COLUNA}' nao possui valores validos.")
    return idades


def calcular_metricas(amostras, media_real, epsilon, tecnica):
    amostras = np.asarray(amostras, dtype=float)
    erros = amostras - media_real
    return {
        'epsilon': epsilon,
        'mse': float(np.mean(erros ** 2)),
    }


def salvar_resultados(resultados, tecnica):
    destino = Path(__file__).resolve().parent / f'resultados-laplace-{tecnica}.csv'
    campos = list(resultados[0])
    with destino.open('w', newline='', encoding='utf-8') as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos, delimiter=';')
        escritor.writeheader()
        escritor.writerows(
            {
                campo: f'{resultado[campo]:.6f}'.replace('.', ',')
                for campo in campos
            }
            for resultado in resultados
        )
    return destino


def imprimir_resultados(resultados):
    print('epsilon;mse')
    for resultado in resultados:
        print(
            f"{resultado['epsilon']:.6f};"
            f"{resultado['mse']:.6f}".replace('.', ',')
        )
