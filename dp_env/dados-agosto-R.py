import os
from pathlib import Path
import numpy as np
import pandas as pd

# Esta instalação do rpy2 inclui o backend ABI, não o backend API compilado.
os.environ.setdefault('RPY2_CFFI_MODE', 'ABI')

import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr
from padrao_laplace import (
    EPSILONS,
    LIMITES,
    N_REPETICOES,
    SEMENTE,
    calcular_metricas,
    carregar_idades,
    imprimir_resultados,
    salvar_resultados,
)

# Ativa a conversão automática entre dados do NumPy/Pandas e R
numpy2ri.activate()
pandas2ri.activate()

# Usa a biblioteca R local do projeto para não exigir instalação global.
r_library = Path(__file__).resolve().parent / 'R' / 'library'
r_library.mkdir(parents=True, exist_ok=True)
existing_r_libraries = list(robjects.r('.libPaths()'))
robjects.r['.libPaths'](
    robjects.StrVector([str(r_library), *existing_r_libraries])
)

diffpriv = importr('diffpriv')

# 1. Carregar os dados no Python
nome_do_arquivo = Path(__file__).resolve().parent / 'heart_disease_cleveland.csv'
if not nome_do_arquivo.exists():
    raise FileNotFoundError(f"O arquivo '{nome_do_arquivo}' não foi encontrado.")

idades_np = carregar_idades()
media_real_idade = float(np.mean(idades_np))
n = len(idades_np)

# 2. Configurar o mecanismo DPMechLaplace do R
sensibilidade = (LIMITES[1] - LIMITES[0]) / n
robjects.r(f'set.seed({SEMENTE})')

# Cria a função alvo em R (calcular a média)
robjects.r('f_media <- function(xs) { mean(xs) }')
f_media = robjects.r['f_media']

# Instancia o objeto DPMechLaplace da lib R diffpriv
mecanismo = diffpriv.DPMechLaplace(
    sensitivity=sensibilidade,
    target=f_media,
    dims=1
)

# 3. Parâmetros de Simulação
resultados = []
for eps in EPSILONS:
    # Instancia a classe DPParamsEps da biblioteca R
    privacy_params = diffpriv.DPParamsEps(epsilon=eps)
    medias_simuladas = []

    for _ in range(N_REPETICOES):
        # Executa o método releaseResponse da lib R
        res = diffpriv.releaseResponse(mecanismo, privacy_params, idades_np)
        
        # Extrai o valor do objeto retornado pelo R
        # O retorno é uma lista R S3/S4, acessamos a resposta pelo índice ou slot
        resposta = float(res.rx2('response')[0])
        medias_simuladas.append(resposta)

    medias_simuladas = np.array(medias_simuladas)
    resultados.append(
        calcular_metricas(medias_simuladas, media_real_idade, eps, 'R-diffpriv')
    )

salvar_resultados(resultados, 'R-diffpriv')
imprimir_resultados(resultados)