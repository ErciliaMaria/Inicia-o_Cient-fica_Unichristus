
import pandas as pd
import numpy as np 

nome_do_arquivo = 'dp_env/diabetes.csv'
    
df = pd.read_csv(nome_do_arquivo)
print("Arquivo 'diabetes.csv' carregado com sucesso!")

media_idade_real = df['Age'].mean()

epsilon = 0.5     
max_idade = 100    
max_idade_2= 90
max_idade_3= 80

sensibilidade = max_idade  
sensibilidade_2 = max_idade_2
sensibilidade_3 = max_idade_3

soma_idade_real = df['Age'].sum()
contagem_idade = df['Age'].count()

b = sensibilidade / epsilon

ruido_laplace = np.random.laplace(loc=0.0, scale=b)
ruido_laplace_2 = np.random.laplace(loc=0.0, scale=(sensibilidade_2 / epsilon))
ruido_laplace_3 = np.random.laplace(loc=0.0, scale=(sensibilidade_3 / epsilon))

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
print(f"1b. Média de Idade das Pacientes (PD com \u03B5={epsilon} e sensibilidade={sensibilidade}): {media_idade_dp:.2f} anos") 
print(f"1c. Média de Idade das Pacientes (PD com \u03B5={epsilon} e sensibilidade={sensibilidade_2}): {media_idade_dp_2:.2f} anos")
print(f"1d. Média de Idade das Pacientes (PD com \u03B5={epsilon} e sensibilidade={sensibilidade_3}): {media_idade_dp_3:.2f} anos")

#  Média de Idade das Pacientes (REAL): 33.24 anos
#  Média de Idade das Pacientes (PD com ε=1.5): 32.95 anos
#  Média de Idade das Pacientes (PD com ε=1.5 e sensibilidade=90): 33.09 anos
#  Média de Idade das Pacientes (PD com ε=1.5 e sensibilidade=80): 33.21 anos
