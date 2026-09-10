from pathlib import Path

import pandas as pd

DIRETORIO = Path(__file__).resolve().parent
ARQUIVOS = {
    'lib-nat': DIRETORIO / 'resultados-laplace-lib-nat.csv',
    'diffprivlib': DIRETORIO / 'resultados-laplace-diffprivlib.csv',
    'R-diffpriv': DIRETORIO / 'resultados-laplace-R-diffpriv.csv',
}

resultados = []
for tecnica, arquivo in ARQUIVOS.items():
    if not arquivo.exists():
        raise FileNotFoundError(
            f"Resultado ausente: {arquivo}. Execute os tres scripts antes."
        )
    resultados.append(pd.read_csv(arquivo))

comparacao = pd.concat(resultados, ignore_index=True)
comparacao = comparacao.sort_values(['epsilon', 'mae', 'mse'])

print(comparacao.to_string(index=False, float_format=lambda valor: f'{valor:.6f}'))
print('\nMelhor tecnica por epsilon (menor MAE):')
melhor_mae = comparacao.loc[comparacao.groupby('epsilon')['mae'].idxmin()]
for _, linha in melhor_mae.sort_values('epsilon').iterrows():
    print(f"epsilon={linha['epsilon']:.6f}: {linha['tecnica']} (MAE={linha['mae']:.6f})")

print('\nMelhor tecnica por epsilon (menor MSE):')
melhor_mse = comparacao.loc[comparacao.groupby('epsilon')['mse'].idxmin()]
for _, linha in melhor_mse.sort_values('epsilon').iterrows():
    print(f"epsilon={linha['epsilon']:.6f}: {linha['tecnica']} (MSE={linha['mse']:.6f})")

saida = DIRETORIO / 'comparacao-laplace.csv'
comparacao.to_csv(saida, index=False)
print(f'\nComparacao salva em: {saida}')
