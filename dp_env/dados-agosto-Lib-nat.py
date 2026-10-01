import random
from padrao_laplace import (
    EPSILONS,
    LIMITES,
    calcular_metricas,
    carregar_idades,
    imprimir_resultados,
    salvar_resultados,
)

idades = carregar_idades()
media_real_idade = float(sum(idades) / len(idades))
sensibilidade = (LIMITES[1] - LIMITES[0]) / len(idades)

def media_privatizada_laplace(dados, epsilon, sensibilidade):
    """
    Calcula a média real e adiciona ruído da distribuição de Laplace.
    Escala do ruído (b) = Sensibilidade / Epsilon
    """
    media_real = sum(dados) / len(dados)
    escala = sensibilidade / epsilon
    
    # Uma Laplace centrada pode ser gerada como exponencial com sinal aleatório.
    magnitude = random.expovariate(1.0 / escala)
    sinal = -1.0 if random.random() < 0.5 else 1.0
    ruido = sinal * magnitude
    
    return media_real + ruido

resultados = []
for epsilon in EPSILONS:
    media_simulada = media_privatizada_laplace(idades, epsilon, sensibilidade)
    resultados.append(
        calcular_metricas([media_simulada], media_real_idade, epsilon, 'lib-nat')
    )

salvar_resultados(resultados, 'lib-nat')
imprimir_resultados(resultados)