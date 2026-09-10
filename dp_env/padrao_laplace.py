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
        'tecnica': tecnica,
        'epsilon': epsilon,
        'media_real': media_real,
        'media_dp': float(np.mean(amostras)),
        'bias': float(np.mean(erros)),
        'mae': float(np.mean(np.abs(erros))),
        'mse': float(np.mean(erros ** 2)),
        'desvio_padrao': float(np.std(amostras)),
        'n_repeticoes': len(amostras),
    }


def salvar_resultados(resultados, tecnica):
    destino = Path(__file__).resolve().parent / f'resultados-laplace-{tecnica}.csv'
    campos = list(resultados[0])
    with destino.open('w', newline='', encoding='utf-8') as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(resultados)
    return destino


def imprimir_resultados(resultados):
    print('tecnica,epsilon,media_real,media_dp,bias,mae,mse,desvio_padrao,n_repeticoes')
    for resultado in resultados:
        print(
            f"{resultado['tecnica']},{resultado['epsilon']:.6f},"
            f"{resultado['media_real']:.6f},{resultado['media_dp']:.6f},"
            f"{resultado['bias']:.6f},{resultado['mae']:.6f},"
            f"{resultado['mse']:.6f},{resultado['desvio_padrao']:.6f},"
            f"{resultado['n_repeticoes']}"
        )
