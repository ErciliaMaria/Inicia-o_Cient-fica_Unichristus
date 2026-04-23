
import pandas as pd
import numpy as np 

nome_do_arquivo = 'dp_env/diabetes.csv'
df = pd.read_csv(nome_do_arquivo)
print("Arquivo 'diabetes.csv' carregado com sucesso!")

media_idade_real = df['Age'].mean()

epsilon = 1.5   
epsilon2 = 1.0
epsilon3 = 0.5

max_idade = 100    

sensibilidade = max_idade  

soma_idade_real = df['Age'].sum()
contagem_idade = df['Age'].count()

b = sensibilidade / epsilon

ruido_laplace = np.random.laplace(loc=0.0, scale=b)
ruido_laplace_2 = np.random.laplace(loc=0.0, scale=(sensibilidade / epsilon2))
ruido_laplace_3 = np.random.laplace(loc=0.0, scale=(sensibilidade / epsilon3))

soma_idade_sem_ruido = soma_idade_real
soma_idade_ruidosa = soma_idade_real + ruido_laplace
soma_idade_ruidosa_2 = soma_idade_real + ruido_laplace_2
soma_idade_ruidosa_3 = soma_idade_real + ruido_laplace_3

media_idade_real = soma_idade_sem_ruido / contagem_idade
media_idade_ruidosa = soma_idade_real + ruido_laplace
media_idade_dp = media_idade_ruidosa / contagem_idade

soma_idade_ruidosa_2 = soma_idade_real + ruido_laplace_2
media_idade_dp_2 = soma_idade_ruidosa_2 / contagem_idade

soma_idade_ruidosa_3 = soma_idade_real + ruido_laplace_3
media_idade_dp_3 = soma_idade_ruidosa_3 / contagem_idade

 
    
print("\n--- Resultados da Análise Descritiva ---")
    
print(f"1a. Média de Idade das Pacientes (REAL): {media_idade_real:.2f} anos") 
print(f"1b. Média de Idade das Pacientes (PD com \u03B5={epsilon}): {media_idade_dp:.2f} anos") 
print(f"1c. Média de Idade das Pacientes (PD com \u03B5={epsilon2}): {media_idade_dp_2:.2f} anos")
print(f"1d. Média de Idade das Pacientes (PD com \u03B5={epsilon3}): {media_idade_dp_3:.2f} anos")

# --- Resultados da Análise Descritiva ---
#  Média de Idade das Pacientes (REAL): 33.24 anos
#  Média de Idade das Pacientes (PD com ε=1.5): 33.16 anos
#  Média de Idade das Pacientes (PD com ε=1.0): 33.14 anos
#  Média de Idade das Pacientes (PD com ε=0.5): 33.37 anos

