from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


nome_do_arquivo = 'dp_env/diabetes_dataset_2.csv'
try:
    df = pd.read_csv(nome_do_arquivo)
    print("Arquivo carregado com sucesso!")
except FileNotFoundError:
    print(f"Erro: O arquivo {nome_do_arquivo} não foi encontrado.")
    exit()

bins = [0, 20, 40, 60, 80]
labels = ['0-20', '21-40', '41-60', '61-80']
df['Faixa_Etaria'] = pd.cut(df['age'], bins=bins, labels=labels, right=False, include_lowest=True)
df = df.dropna(subset=['Faixa_Etaria'])

epsilons = [0.1, np.log(2), np.log(4), 1.0]
epsilon_labels = ['0.1', 'ln(2)', 'ln(4)', '1.0']
contagem_real = df['Faixa_Etaria'].value_counts().reindex(labels, fill_value=0)


def aplicar_exponencial_local(valor_real, eps, cats):
    peso_acerto = np.exp(eps / 2)
    peso_erro = 1.0
    probs = np.array([peso_acerto if cat == valor_real else peso_erro for cat in cats], dtype=float)
    probs = probs / probs.sum()
    return np.random.choice(cats, p=probs)


x = np.arange(len(labels))
largura = 0.35
fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
axs = axs.flatten()
relatorio_erros = []

for ax, epsilon, epsilon_label in zip(axs, epsilons, epsilon_labels):
    faixa_privada = df['Faixa_Etaria'].apply(lambda valor: aplicar_exponencial_local(valor, epsilon, labels))
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

for ax in axs[::2]:
    ax.set_ylabel('Número de Pacientes')
for ax in axs[2:]:
    ax.set_xlabel('Faixas Etárias')

fig.suptitle('Privacidade Diferencial Local: Real vs Mecanismo Exponencial', fontsize=14)
plt.tight_layout()
plt.savefig('resultado_exponencial_completo2.png', dpi=300)

print("\n--- Relatório ---")
for epsilon_label, erro_absoluto, mae in relatorio_erros:
    print(f"epsilon={epsilon_label} -> Erro Absoluto Total: {erro_absoluto:.0f} | MAE: {mae:.2f}")
print("Gráfico gerado: 'resultado_exponencial_completo2.png'")