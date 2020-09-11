#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sk
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler


# In[2]:


from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[3]:


countries = pd.read_csv("countries.csv")


# In[4]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[5]:


countries.columns


# In[6]:


columns_to_parse_to_float = ['Agriculture', 'Arable', 'Birthrate', 'Climate',
                             'Coastline_ratio', 'Crops', 'Deathrate', 'Industry',
                             'Infant_mortality', 'Literacy', 'Net_migration', 'Other',
                             'Phones_per_1000', 'Pop_density', 'Service']
for column in columns_to_parse_to_float:
    countries[column] = pd.to_numeric(countries[column].str.replace(',', '.'), errors='coerce')
countries['Region'] = countries['Region'].str.strip()
countries['Country'] = countries['Country'].str.strip()


# In[7]:


countries.dtypes


# In[8]:


countries.isnull().sum()


# In[9]:


sns.boxplot(x=countries["Net_migration"])


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[10]:


def q1():
    return list(np.sort(countries['Region'].unique()))


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[11]:


def q2():
    X = np.array(countries['Pop_density']).reshape(-1,1)
    y = countries['Region']
    estimator = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    result = estimator.fit_transform(X, y) >= 9.
    return int(np.sum(result))


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[12]:


def q3():
    enc = OneHotEncoder(handle_unknown='ignore')
    new_dt = pd.DataFrame(countries[['Region', 'Climate']])
    new_dt.Climate.fillna(0, inplace=True)
    result = enc.fit_transform(new_dt)
    return int(result.shape[1])


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[13]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]

columns = countries.select_dtypes(include=['int64', 'float64']).columns

pipeline = Pipeline(steps=[('imp', SimpleImputer(strategy='median')),
                           ('scaler', StandardScaler())])
transformer = ColumnTransformer(transformers=[('number', pipeline, columns)], n_jobs=-1)

transformer.fit(countries)


# In[14]:


def q4():
    test_dt = pd.DataFrame([test_country], columns=countries.columns)
    result = transformer.transform(test_dt)[0][columns.get_loc('Arable')]
    return round(float(result),3)


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[15]:


def q5():
    q1, q3 = countries['Net_migration'].dropna().quantile([.25, .75])
    iqr = q3 - q1

    inferior_outlier = q1 - 1.5 * iqr
    upper_outlier = q3 + 1.5 * iqr

    outliers_below = int(countries[countries['Net_migration'] < inferior_outlier].shape[0])
    outliers_above = int(countries[countries['Net_migration'] > upper_outlier].shape[0])

    return outliers_below, outliers_above, False


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[16]:


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=12)


# In[17]:


def q6():
    vectorizer = CountVectorizer()
    document_term_matrix = vectorizer.fit_transform(newsgroup.data)
    map_vocabulary = vectorizer.vocabulary_
    return int(document_term_matrix[:, map_vocabulary['phone']].sum())


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[18]:


def q7():
    tfidf_vectorizer = TfidfVectorizer().fit(newsgroup.data)
    tfidf_count = tfidf_vectorizer.transform(newsgroup.data)
    map_vocabulary = tfidf_vectorizer.vocabulary_
    return float(tfidf_count[:, map_vocabulary['phone']].sum().round(3))

