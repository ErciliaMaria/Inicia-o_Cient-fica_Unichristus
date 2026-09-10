from diffprivlib import tools
import numpy as np
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

idades = carregar_idades()
media_real_idade = float(np.mean(idades))
np.random.seed(SEMENTE)

resultados = []
for epsilon in EPSILONS:
	medias_simuladas = np.array([
		tools.mean(idades, epsilon=epsilon, bounds=LIMITES)
		for _ in range(N_REPETICOES)
	], dtype=float)
	resultados.append(
		calcular_metricas(medias_simuladas, media_real_idade, epsilon, 'diffprivlib')
	)

salvar_resultados(resultados, 'diffprivlib')
imprimir_resultados(resultados)