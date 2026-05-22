from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


nome_do_arquivo = 'dp_env/diabetes.csv'
try:
    df = pd.read_csv(nome_do_arquivo)
    print("Arquivo carregado com sucesso!")
except FileNotFoundError:
    print(f"Erro: O arquivo {nome_do_arquivo} não foi encontrado.")
    exit()

# 1. Padroniza a idade em faixas iguais às demais técnicas.
bins = [0, 20, 40, 60, 80]
labels = ['0-20', '21-40', '41-60', '61-80']
df['Faixa_Etaria'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False, include_lowest=True)
df = df.dropna(subset=['Faixa_Etaria'])

# 2. Define os budgets de privacidade que serão comparados.
epsilons = [0.1, np.log(2), np.log(4), 1.0]
epsilon_labels = ['0.1', 'ln(2)', 'ln(4)', '1.0']
# Contagem real usada como referência para calcular o erro.
contagem_real = df['Faixa_Etaria'].value_counts().reindex(labels, fill_value=0)


# 3. Mapeia cada faixa etária para categorias numéricas de 1 a 4.
categoria_id = {label: i + 1 for i, label in enumerate(labels)}


def funcao_score(cat_real_id, cat_saida_id):
    # Penalidade quadrática negativa: maior distância entre categorias => menor escore.
    return -float((cat_saida_id - cat_real_id) ** 2)


def calcular_sensibilidade_global(ids):
    """
    Calcula Delta_u global uma única vez considerando:
    u(c_real, c_saida) = -(c_saida - c_real)^2
    """
    max_diff = 0.0
    for c_real in ids:
        for c_real_linha in ids:
            for c_saida in ids:
                diff = abs(funcao_score(c_real, c_saida) - funcao_score(c_real_linha, c_saida))
                if diff > max_diff:
                    max_diff = diff
    return float(max_diff)


# Sensibilidade global fixa para todo o experimento.
delta_u_global = calcular_sensibilidade_global(list(categoria_id.values()))


def aplicar_exponencial_local(valor_real, eps, cats, cat_to_id, delta_u):
    cat_real_id = cat_to_id[valor_real]

    # 4. Calcula os escores quadráticos negativos para cada categoria candidata.
    scores = np.array([
        funcao_score(cat_real_id, cat_to_id[categoria])
        for categoria in cats
    ], dtype=float)

    # Se a sensibilidade for zero, escolhe deterministicamente o melhor escore.
    if delta_u == 0:
        idx = int(np.argmax(scores))
        return cats[idx]

    pesos = np.exp((eps * scores) / (2.0 * delta_u))
    probs = pesos / pesos.sum()
    return np.random.choice(cats, p=probs)


# 6. Prepara a figura e as posições das barras.
x = np.arange(len(labels))
largura = 0.35
fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
axs = axs.flatten()
relatorio_erros = []

# 7. Para cada epsilon, gera uma versão privada da contagem e mede o MAE.
for ax, epsilon, epsilon_label in zip(axs, epsilons, epsilon_labels):
    faixa_privada = df['Faixa_Etaria'].apply(
        lambda valor: aplicar_exponencial_local(valor, epsilon, labels, categoria_id, delta_u_global)
    )
    contagem_exponencial = faixa_privada.value_counts().reindex(labels, fill_value=0)

    erro_absoluto = np.abs(contagem_real.values - contagem_exponencial.values).sum()
    mae = np.mean(np.abs(contagem_real.values - contagem_exponencial.values))
    relatorio_erros.append((epsilon_label, erro_absoluto, mae))

    barras_real = ax.bar(x - largura / 2, contagem_real.values, largura, label='Real (Original)', color='#2c3e50', alpha=0.9)
    barras_exp = ax.bar(x + largura / 2, contagem_exponencial.values, largura, label=rf'Mec. Exponencial Local ($\epsilon$={epsilon_label})', color='#d35400', alpha=0.9)

    ax.set_title(rf'Comparação com $\epsilon$={epsilon_label}\nMAE: {mae:.2f} | Erro: {erro_absoluto:.0f}', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=8)
    ax.bar_label(barras_real, padding=2, fontsize=8)
    ax.bar_label(barras_exp, padding=2, fontsize=8)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.text(
        0.03,
        0.95,
        f'MAE: {mae:.2f}',
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=9,
        fontweight='bold',
        bbox={"facecolor": "white", "alpha": 0.8, "pad": 2}
    )

# 8. Ajusta rótulos e título do gráfico final.
for ax in axs[::2]:
    ax.set_ylabel('Número de Pacientes')
for ax in axs[2:]:
    ax.set_xlabel('Faixas Etárias')

fig.suptitle('Privacidade Diferencial Local: Real vs Mecanismo Exponencial', fontsize=14)
plt.tight_layout()
plt.savefig('resultado_exponencial_completo.png', dpi=300)

# 9. Exibe o relatório de erro por epsilon.
print("\n--- Relatório ---")
print(f"Delta_u global: {delta_u_global:.2f}")
for epsilon_label, erro_absoluto, mae in relatorio_erros:
    print(f"epsilon={epsilon_label} -> Erro Absoluto Total: {erro_absoluto:.0f} | MAE: {mae:.2f}")
print("Gráfico gerado: 'resultado_exponencial_completo.png'")