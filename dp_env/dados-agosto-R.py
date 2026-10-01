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
existing_r_libraries = []
for library in robjects.r('.libPaths()'):
    library_path = Path(str(library))
    if library_path.is_dir() and any(
        (package / 'DESCRIPTION').is_file() for package in library_path.iterdir()
    ):
        existing_r_libraries.append(str(library_path))
robjects.r['.libPaths'](
    robjects.StrVector([str(r_library), *existing_r_libraries])
)

diffpriv = importr('diffpriv', lib_loc=str(r_library))

# 1. Carregar os dados no Python
nome_do_arquivo = Path(__file__).resolve().parent / 'heart_disease_cleveland.csv'
if not nome_do_arquivo.exists():
    raise FileNotFoundError(f"O arquivo '{nome_do_arquivo}' não foi encontrado.")

idades_np = carregar_idades()
media_real_idade = float(np.mean(idades_np))
n = len(idades_np)

# 2. Configurar o mecanismo DPMechLaplace do R
sensibilidade = (LIMITES[1] - LIMITES[0]) / n

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

    # Executa uma única rodada do mecanismo R e preserva uma amostra de tamanho 1.
    res = diffpriv.releaseResponse(mecanismo, privacy_params, idades_np)

    # O retorno é uma lista R S3/S4; a resposta fica no campo response.
    resposta = float(res.rx2('response')[0])
    medias_simuladas = np.array([resposta])
    resultados.append(
        calcular_metricas(medias_simuladas, media_real_idade, eps, 'R-diffpriv')
    )

salvar_resultados(resultados, 'R-diffpriv')
imprimir_resultados(resultados)