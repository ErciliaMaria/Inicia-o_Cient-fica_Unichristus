import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

nome_do_arquivo = 'dp_env/diabetes.csv'
try:
    df = pd.read_csv(nome_do_arquivo)
    print("Arquivo carregado com sucesso!")
except FileNotFoundError:
    print(f"Erro: O arquivo {nome_do_arquivo} não foi encontrado.")
    exit()

bins = [0, 20, 40, 60, 80]
labels = ['0-20', '20-40', '40-60', '60-80']
df['Faixa_Etaria'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# Remove possíveis valores nulos fora das faixas definidas
df = df.dropna(subset=['Faixa_Etaria'])

# 3. Parâmetros de Privacidade (Mecanismo Exponencial Local)
epsilons = [0.1, np.log(2), np.log(4), 1.0]
epsilon_labels = ['0.1', 'ln(2)', 'ln(4)', '1.0']
categorias = labels
k = len(categorias)

def aplicar_exponencial_local(valor_real, eps, cats):
  
    peso_acerto = np.exp(eps * 1 / 2)
    peso_erro = np.exp(eps * 0 / 2) # Sempre será 1
    
    probs = []
    for cat in cats:
        if cat == valor_real:
            probs.append(peso_acerto)
        else:
            probs.append(peso_erro)
    
    # Normalização para que a soma das probabilidades seja 1
    probs = np.array(probs) / sum(probs)
    
    # Sorteio aleatório baseado nas probabilidades
    return np.random.choice(cats, p=probs)

# 4. Processamento dos Dados
# Contagem Real: garantindo que as 4 categorias apareçam (mesmo se vazias)
contagem_real = df['Faixa_Etaria'].value_counts().reindex(labels, fill_value=0)

# 5. Visualização com Matplotlib
x = np.arange(len(labels))
largura = 0.35

fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
axs = axs.flatten()

relatorio_erros = []

for ax, epsilon, epsilon_label in zip(axs, epsilons, epsilon_labels):
    faixa_privada = df['Faixa_Etaria'].apply(lambda valor: aplicar_exponencial_local(valor, epsilon, labels))
    contagem_exponencial = faixa_privada.value_counts().reindex(labels, fill_value=0)

    erro_absoluto = np.abs(contagem_real - contagem_exponencial).sum()
    mae = np.mean(np.abs(contagem_real - contagem_exponencial))
    relatorio_erros.append((epsilon_label, erro_absoluto, mae))

    barras_real = ax.bar(
        x - largura/2,
        contagem_real.values,
        largura,
        label='Real (Original)',
        color='#2c3e50',
        alpha=0.9
    )
    barras_exp = ax.bar(
        x + largura/2,
        contagem_exponencial.values,
        largura,
        label=rf'Mec. Exponencial Local ($\epsilon$={epsilon_label})',
        color='#d35400',
        alpha=0.9
    )

    ax.set_title(
        rf'Comparação com $\epsilon$={epsilon_label}\nMAE: {mae:.2f} | Erro: {erro_absoluto:.0f}',
        fontsize=11
    )
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
    ax.set_ylabel('Número de Pacientes', fontsize=11)
for ax in axs[2:]:
    ax.set_xlabel('Faixas Etárias', fontsize=11)

fig.suptitle('Privacidade Diferencial Local: Real vs Mecanismo Exponencial', fontsize=14)
plt.tight_layout()

plt.savefig('resultado_exponencial_completo.png', dpi=300)
print(f"\n--- Relatório ---")
for epsilon_label, erro_absoluto, mae in relatorio_erros:
    print(f"epsilon={epsilon_label} -> Erro Absoluto Total: {erro_absoluto:.0f} | MAE: {mae:.2f}")
print(f"Gráfico gerado: 'resultado_exponencial_completo.png'")