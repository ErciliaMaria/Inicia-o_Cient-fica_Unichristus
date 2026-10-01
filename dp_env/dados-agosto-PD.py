import diffprivlib as dp
from diffprivlib import tools
import numpy as np
from padrao_laplace import (
	EPSILONS,
	LIMITES,
	calcular_metricas,
	carregar_idades,
	imprimir_resultados,
	salvar_resultados,
)

idades = carregar_idades()

media_real_idade = float(np.mean(idades))

resultados = []
for epsilon in EPSILONS:
	media_simulada = tools.mean(idades, epsilon=epsilon, bounds=LIMITES)
	resultados.append(
		calcular_metricas([media_simulada], media_real_idade, epsilon, 'diffprivlib')
	)

salvar_resultados(resultados, 'diffprivlib')
imprimir_resultados(resultados)