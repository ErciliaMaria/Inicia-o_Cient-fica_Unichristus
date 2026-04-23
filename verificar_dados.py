import pandas as pd

from ydata_profiling import ProfileReport

nome_do_arquivo = 'dp_env/diabetes.csv'

df = pd.read_csv(nome_do_arquivo)
profile = ProfileReport(df, title='Relatório de Análise de Dados', explorative=True)
profile.to_file('relatorio.html')
print("Arquivo carregado com sucesso!")
    
