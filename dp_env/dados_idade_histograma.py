from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 1. Carregamento e Preparação (Seu código base)
nome_do_arquivo = 'dp_env/diabetes.csv'
df = pd.read_csv(nome_do_arquivo)

bins = [0, 20, 40, 60, 80, 100]
labels = ['0-19', '20-39', '40-59', '60-79', '80-99']
df['Faixa_Etaria'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False, include_lowest=True)
df = df.dropna(subset=['Faixa_Etaria'])

epsilons = [0.1, np.log(2), np.log(4), 1.0]
epsilon_labels = ['0.1', 'ln(2)', 'ln(4)', '1.0']
contagem_real = df['Faixa_Etaria'].value_counts().reindex(labels, fill_value=0)

x = np.arange(len(labels))
largura = 0.35
fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
axs = axs.flatten()
relatorio_erros = []

for ax, epsilon, epsilon_label in zip(axs, epsilons, epsilon_labels):
    ruido = np.random.laplace(loc=0.0, scale=1.0 / epsilon, size=len(labels))
    contagem_dp = np.maximum(contagem_real.values + ruido, 0)

    erro_absoluto = np.abs(contagem_real.values - contagem_dp).sum()
    mae = np.mean(np.abs(contagem_real.values - contagem_dp))
    relatorio_erros.append((epsilon_label, erro_absoluto, mae))

    barras_real = ax.bar(x - largura / 2, contagem_real.values, largura, label='Real', color='#34495e')
    barras_dp = ax.bar(x + largura / 2, contagem_dp, largura, label=rf'Laplace ($\epsilon$={epsilon_label})', color='#3498db')

    ax.set_title(rf'Comparação com $\epsilon$={epsilon_label}\nMAE: {mae:.2f} | Erro: {erro_absoluto:.0f}', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.bar_label(barras_real, padding=2, fmt='%.0f', fontsize=8)
    ax.bar_label(barras_dp, padding=2, fmt='%.0f', fontsize=8)
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

for ax in axs[::2]:
    ax.set_ylabel('Número de Pacientes')

fig.suptitle('Privacidade Diferencial (Laplace) por Faixa Etária', fontsize=14)
plt.tight_layout()
plt.savefig('resultado_privacidade.png', dpi=300)

print("Gráfico salvo como 'resultado_privacidade.png'")
print("\n--- Relatório MAE por epsilon ---")
for epsilon_label, erro_absoluto, mae in relatorio_erros:
    print(f"epsilon={epsilon_label} -> Erro Absoluto Total: {erro_absoluto:.0f} | MAE: {mae:.2f}")