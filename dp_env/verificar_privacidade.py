from tkinter.messagebox import IGNORE
import pandas as pd
import numpy as np 

nome_do_arquivo = 'dp_env/diabetes.csv'

try:
    
    df = pd.read_csv(nome_do_arquivo)
    print("Arquivo 'diabetes.csv' carregado com sucesso!")

    media_idade_real = df['Age'].mean()

    epsilon = 1.5     
    max_idade = 100    
    sensibilidade = max_idade  

    soma_idade_real = df['Age'].sum()
    contagem_idade = df['Age'].count()

    b = sensibilidade / epsilon

    ruido_laplace = np.random.laplace(loc=0.0, scale=b)

    soma_idade_ruidosa = soma_idade_real + ruido_laplace
    media_idade_dp = soma_idade_ruidosa / contagem_idade

    colunas_para_analise = [
        'Pregnancies',
        'Glucose',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
        'BMI',
        'DiabetesPedigreeFunction'
    ]

    medias_gerais = df[colunas_para_analise].mean()
    
    print("\n--- Resultados da Análise Descritiva ---")
    
    print(f"1a. Média de Idade das Pacientes (REAL): {media_idade_real:.2f} anos") --- 33.24 ---
    print(f"1b. Média de Idade das Pacientes (PD com \u03B5={epsilon}): {media_idade_dp:.2f} anos") --- 32.93 ---

    
    print("\n2. Médias das demais características (SEM PD):")
    print(medias_gerais.apply(lambda x: f'{x:.2f}').to_string())

except FileNotFoundError:
    print(f"Erro: O arquivo '{nome_do_arquivo}' não foi encontrado. Verifique o caminho e o nome do arquivo.")
except ImportError:
    print("Erro: A biblioteca 'numpy' é necessária para a Privacidade Diferencial. Instale-a com 'pip install numpy'.")
except Exception as e:
    print(f"Ocorreu um erro ao processar o arquivo: {e}")