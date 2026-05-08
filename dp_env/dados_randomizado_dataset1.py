from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 1. Configurações e Carregamento
nome_do_arquivo = 'dp_env/diabetes.csv'
df = pd.read_csv(nome_do_arquivo)

bins = [0, 20, 40, 60, 80, 100]
labels = ['0-19', '20-39', '40-59', '60-79', '80-99']
df['Faixa_Etaria'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False, include_lowest=True)
df = df.dropna(subset=['Faixa_Etaria'])

epsilons = [0.1, np.log(2), np.log(4), 1.0]
epsilon_labels = ['0.1', 'ln(2)', 'ln(4)', '1.0']
contagem_real = df['Faixa_Etaria'].value_counts().reindex(labels, fill_value=0)


def perturbacao_unaria(valor_real, categorias, p, q):
    vetor = np.zeros(len(categorias))
    idx_real = categorias.index(valor_real)
    vetor[idx_real] = 1

    vetor_perturbado = []
    for bit in vetor:
        if bit == 1:
            novo_bit = 1 if np.random.rand() < p else 0
        else:
            novo_bit = 1 if np.random.rand() < q else 0
        vetor_perturbado.append(novo_bit)

    return np.array(vetor_perturbado)


x = np.arange(len(labels))
largura = 0.35
fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
axs = axs.flatten()
relatorio_erros = []
n = len(df)

for ax, epsilon, epsilon_label in zip(axs, epsilons, epsilon_labels):
    p = np.exp(epsilon / 2) / (np.exp(epsilon / 2) + 1)
    q = 1 / (np.exp(epsilon / 2) + 1)

    respostas_perturbadas = df['Faixa_Etaria'].astype(str).apply(lambda valor: perturbacao_unaria(valor, labels, p, q))
    soma_vetores = np.sum(respostas_perturbadas.values, axis=0)
    contagem_corrigida = (soma_vetores - n * q) / (p - q)
    contagem_corrigida = np.maximum(contagem_corrigida, 0)

    erro_absoluto = np.abs(contagem_real.values - contagem_corrigida).sum()
    mae = np.mean(np.abs(contagem_real.values - contagem_corrigida))
    relatorio_erros.append((epsilon_label, erro_absoluto, mae))

    barras_real = ax.bar(x - largura / 2, contagem_real.values, largura, label='Real', color='#2c3e50')
    barras_ldp = ax.bar(x + largura / 2, contagem_corrigida, largura, label=rf'Unary Encoding ($\epsilon$={epsilon_label})', color='#27ae60')

    ax.set_title(rf'Comparação com $\epsilon$={epsilon_label}\nMAE: {mae:.2f} | Erro: {erro_absoluto:.0f}', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=8)
    ax.bar_label(barras_real, padding=2, fontsize=8)
    ax.bar_label(barras_ldp, padding=2, fmt='%.0f', fontsize=8)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.text(
        0.5,
        0.02,
        f'Erro: {erro_absoluto:.0f} | MAE: {mae:.2f}',
        transform=ax.transAxes,
        ha='center',
        va='bottom',
        fontsize=8,
        bbox={"facecolor": "green", "alpha": 0.08, "pad": 2}
    )

for ax in axs[::2]:
    ax.set_ylabel('Número de Pacientes')

fig.suptitle('Privacidade Local: Real vs Randomized Response (Unary Encoding)', fontsize=14)
handles, labels_legend = axs[0].get_legend_handles_labels()
fig.legend(handles, labels_legend, loc='upper center', ncol=2, frameon=False)

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.savefig('resultado_unary_encoding.png')
if plt.isinteractive():
    plt.show()
else:
    plt.close(fig)

print("\n--- Relatório ---")
for epsilon_label, erro_absoluto, mae in relatorio_erros:
    print(f"epsilon={epsilon_label} -> Erro Absoluto Total: {erro_absoluto:.0f} | MAE: {mae:.2f}")
print("Gráfico gerado com sucesso!")