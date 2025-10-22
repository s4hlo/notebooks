# %% [markdown]
# ### Instruções Gerais
# 
# Siga os passos abaixo como um guia para estruturar sua análise. O objetivo não é apenas gerar gráficos, mas também interpretá-los para entender a história que os dados contam.
# 
# #### Passo 1: Preparação do Ambiente e Carregamento dos Dados
# 
# Comece criando seu notebook e preparando o ambiente para a análise.
# 
# 1.  Crie um novo notebook no Google Colab ou em seu ambiente Jupyter local.
# 2.  Importe as bibliotecas essenciais para a análise. No início do seu código, inclua:
#     ```python
#     import pandas as pd
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#     ```
# 3.  Carregue o dataset. Primeiro, baixe o arquivo `.csv` do [Kaggle](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors/data) para o mesmo ambiente do seu notebook. Em seguida, use o pandas para carregá-lo em um DataFrame:
#     ```python
#     # Substitua 'caminho/para/seu/arquivo.csv' pelo caminho correto do arquivo
#     df = pd.read_csv('caminho/para/seu/arquivo.csv')
#     ```
# 4.  Inspecione o início do seu DataFrame para garantir que os dados foram carregados corretamente, utilizando o comando `df.head()`.
# 
# #### Passo 2: Exploração Inicial e Limpeza
# 
# Antes de criar visualizações, é fundamental entender a estrutura e a qualidade dos seus dados.
# 
# 1.  Verifique as informações gerais do DataFrame usando `df.info()`. Observe os tipos de dados de cada coluna (são numéricos? categóricos?) e se há valores nulos.
# 2.  Conte os dados faltantes (nulos) em cada coluna com o comando `df.isnull().sum()`. Se houver dados faltantes, será necessário um tratamento (descartar, substituir pela média?).
# 3.  Gere estatísticas descritivas para as colunas numéricas usando `df.describe()`. Isso te dará uma ideia da média, desvio padrão, valores mínimos e máximos de variáveis como `Exam_Score` e `Hours_Studied`.
# 
# #### Passo 3: Análise e Visualização dos Dados
# 
# Agora é a hora de criar gráficos para responder perguntas e encontrar padrões. Explore os dados com os seguintes tipos de análise:
# 
# 1.  Analise a distribuição da variável mais importante, `Exam_Score`.
#     * Crie um histograma com `seaborn.histplot()` para ver como as notas dos alunos estão distribuídas. Elas se concentram em alguma faixa específica?
# 
# 2.  Investigue as variáveis categóricas
#     * Conte a quantidade de alunos por categoria em colunas como `School_Type`, `Parental_Involvement` e `Gender` usando `df['nome_da_coluna'].value_counts()`.
#     * Crie um gráfico de barras (`seaborn.countplot()`) para visualizar essas contagens para pelo menos duas variáveis categóricas de sua escolha.
# 
# 3.  Explore a relação entre duas variáveis
#     * Compare a distribuição de `Exam_Score` para diferentes categorias. Use um boxplot (`seaborn.boxplot()`) para comparar as notas entre `Gender` (gênero) ou entre `School_Type` (tipo de escola).
#     * Investigue a relação entre `Hours_Studied` e `Exam_Score`. Crie um gráfico de dispersão (`seaborn.scatterplot()`) para ver se há uma tendência visível.
# 
# 4.  Analise a correlação entre todas as variáveis numéricas.
#     * Calcule a matriz de correlação. Dica: selecione apenas as colunas numéricas antes de usar o método `.corr()`.
#     * Crie um mapa de calor (heatmap) com `seaborn.heatmap()` para visualizar a matriz de correlação. Isso facilita a identificação das relações mais fortes (positivas ou negativas) entre as variáveis.

# %% [markdown]
# # ANÁLISE EXPLORATÓRIA DE DADOS
# 
# Seu nome.

# %%
# Inicie seu código aqui...

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar o dataset
df = pd.read_csv('../secret/StudentPerformanceFactors.csv')
print("Primeiras 5 linhas do dataset:")
df.head()

# %%
# Passo 2: Exploração Inicial e Limpeza

# 1. Verificar informações gerais do DataFrame
print("Informações gerais do DataFrame:")
df.info()

# %%
# 2. Verificar dados faltantes
print("\nDados faltantes por coluna:")
print(df.isnull().sum())

# %%
# 3. Estatísticas descritivas para colunas numéricas
print("\nEstatísticas descritivas:")
df.describe()
# %%
print("Shape do DataFrame:")
df.shape

# %%
# Passo 3: Análise e Visualização dos Dados

# 1. Análise da distribuição da variável Exam_Score
print("Análise da distribuição de Exam_Score:")
print(f"Média: {df['Exam_Score'].mean():.2f}")
print(f"Mediana: {df['Exam_Score'].median():.2f}")
print(f"Desvio padrão: {df['Exam_Score'].std():.2f}")

# Histograma da distribuição de Exam_Score
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Exam_Score', bins=20, kde=True)
plt.title('Distribuição das Pontuações no Exame')
plt.xlabel('Pontuação no Exame')
plt.ylabel('Frequência')
plt.show()

# %%
# Boxplot para visualizar a distribuição de Exam_Score
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Exam_Score')
plt.title('Boxplot - Distribuição das Pontuações no Exame')
plt.xlabel('Pontuação no Exame')
plt.show()

# %%
# 2. Análise de variáveis categóricas

# Contagem de alunos por categoria
print("Contagem por School_Type:")
print(df['School_Type'].value_counts())

print("\nContagem por Gender:")
print(df['Gender'].value_counts())

print("\nContagem por Parental_Involvement:")
print(df['Parental_Involvement'].value_counts())

# Gráficos de barras para variáveis categóricas
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sns.countplot(data=df, x='School_Type', ax=axes[0])
axes[0].set_title('Distribuição por Tipo de Escola')
axes[0].tick_params(axis='x', rotation=45)

sns.countplot(data=df, x='Gender', ax=axes[1])
axes[1].set_title('Distribuição por Gênero')

sns.countplot(data=df, x='Parental_Involvement', ax=axes[2])
axes[2].set_title('Distribuição por Envolvimento dos Pais')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# %%
# 3. Exploração de relações entre variáveis

# Boxplot comparando Exam_Score por School_Type e Gender
plt.figure(figsize=(12, 8))
sns.boxplot(data=df, x='School_Type', y='Exam_Score', hue='Gender')
plt.title('Distribuição de Pontuações por Tipo de Escola e Gênero')
plt.xlabel('Tipo de Escola')
plt.ylabel('Pontuação no Exame')
plt.legend(title='Gênero')
plt.xticks(rotation=45)
plt.show()

# %%
# Gráfico de dispersão: Hours_Studied vs Exam_Score
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Hours_Studied', y='Exam_Score', alpha=0.6)
plt.title('Relação entre Horas de Estudo e Pontuação no Exame')
plt.xlabel('Horas de Estudo')
plt.ylabel('Pontuação no Exame')
plt.show()

# %%
# Preparação dos dados para análise de correlação
# Convertendo variáveis categóricas para numéricas
df_numeric = df.copy()
df_numeric = df_numeric.replace({"Low":0, "Medium": 1, "High": 2,
                                 "No": -1, "Yes":1,
                                 "Near":0, "Moderate": 1, 'Far': 2,
                                 "High School": 0, "College": 1, "Postgraduate": 2,
                                 'Positive': 1, 'Neutral': 0, 'Negative':-1})

print("Dados convertidos para análise numérica:")
print("Tipos de dados após conversão:")
print(df_numeric.dtypes)

# %%
# 4. Análise de correlação entre variáveis numéricas

# Verificar valores únicos de Peer_Influence para conversão
print("Valores únicos de Peer_Influence:")
print(df['Peer_Influence'].unique())

# Converter Peer_Influence também
df_numeric['Peer_Influence'] = df_numeric['Peer_Influence'].replace({
    'Positive': 1, 'Neutral': 0, 'Negative': -1
})

print("\nTipos de dados finais:")
print(df_numeric.dtypes)

# %%
# Matriz de correlação e heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = df_numeric.corr(numeric_only=True).round(2)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": .8})
plt.title('Matriz de Correlação entre Variáveis')
plt.tight_layout()
plt.show()

# Mostrar correlações com Exam_Score
print("\nCorrelações com Exam_Score (ordenadas por valor absoluto):")
exam_correlations = df_numeric.corr(numeric_only=True)['Exam_Score'].abs().sort_values(ascending=False)
print(exam_correlations)

# %%
# Análise adicional: Pairplot para visualizar todas as relações
plt.figure(figsize=(15, 12))
sns.pairplot(df_numeric[['Exam_Score', 'Hours_Studied', 'Family_Income', 'Parental_Involvement', 'Extracurricular_Activities']], 
             diag_kind='hist', plot_kws={'alpha': 0.6})
plt.suptitle('Análise de Pares de Variáveis', y=1.02)
plt.show()

# %%
# Análise de regressão linear: Hours_Studied vs Exam_Score por Family_Income
plt.figure(figsize=(12, 8))
sns.lmplot(data=df, x='Hours_Studied', y='Exam_Score', hue='Family_Income', 
           height=6, aspect=1.2)
plt.suptitle('Relação entre Horas de Estudo e Pontuação por Renda Familiar', y=1.02)
plt.show()

# %%
# Análise de atividades extracurriculares vs desempenho
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Extracurricular_Activities', y='Exam_Score')
plt.title('Distribuição de Pontuações por Participação em Atividades Extracurriculares')
plt.xlabel('Participa em Atividades Extracurriculares')
plt.ylabel('Pontuação no Exame')
plt.show()

# %%
# Análise de envolvimento dos pais vs desempenho
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Parental_Involvement', y='Exam_Score')
plt.title('Distribuição de Pontuações por Envolvimento dos Pais')
plt.xlabel('Nível de Envolvimento dos Pais')
plt.ylabel('Pontuação no Exame')
plt.xticks(rotation=45)
plt.show()

# %% [markdown]
# # PERGUNTAS
# 
# Após realizar as análises e gerar os gráficos, utilize células de texto (Markdown) no seu notebook para responder às seguintes perguntas. Justifique cada resposta com base nos gráficos e dados que você analisou.
# 
# 1. Com base no mapa de calor (heatmap) de correlação e em outros gráficos que você gerou, quais são os 2 ou 3 fatores que parecem ter a correlação mais forte e positiva com a pontuação final no exame (Exam_Score)? Justifique sua resposta mencionando os gráficos que te levaram a essa conclusão.
# 
# 2. Existe uma diferença clara no desempenho (Exam_Score) entre alunos que participam de atividades extracurriculares (Extracurricular_Activities) e os que não participam? E entre os diferentes níveis de envolvimento dos pais (Parental_Involvement)? Use os boxplots ou outros gráficos para explicar sua conclusão.
# 
# 3. Como as pontuações dos exames (Exam_Score) estão distribuídas? Elas se concentram em uma faixa específica (por exemplo, notas altas, médias ou baixas)? O que o histograma que você criou revela sobre o desempenho geral dos alunos neste dataset?
# 
# 4. Analisando o gráfico de dispersão, qual é a relação entre as Hours_Studied (Horas de Estudo) e a Exam_Score (Pontuação no Exame)? Um aumento nas horas de estudo parece garantir uma nota maior? Explique o que o padrão dos pontos no gráfico sugere.
# 
# 5. Além das correlações mais óbvias, qual outra variável lhe chamou a atenção pela sua aparente influência no desempenho dos alunos? Apresente um gráfico que suporte sua observação e descreva o insight que você obteve ao analisar essa relação.

# %% [markdown]
# << SUA RESPOSTA EM FORMATO MARKDOWN >>


