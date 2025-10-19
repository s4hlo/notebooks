# %%
# Importando as bibliotecas

import numpy as np
import pandas as pd
from sklearn.svm import SVC

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
data = pd.read_csv(url, delimiter=",", header=None)
header = ["age", "sex", "cp", "trestpbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
data.columns = header
data

# selecionando os dados
X = data.iloc[:,0:13]
X

# selecionando a target
y = data.iloc[:,13]

# valores maiores que 1 foram transformados em valor 1
y = [1 if x > 0 else 0 for x in y]
y

# encontrar valores faltantes
X[X.values == '?'].index

# remover linhas com valores faltantes nos dados
id2drop = X[X.values == '?'].index
X = X.drop(id2drop)

# remover linhas com valores faltantes no target
y = pd.Series(y).drop(id2drop)
y
# %%

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

"""## Treinando um modelo

Vamos treinar um classificador SVM com os valores padrões do hiperparâmetro e verificar a performance do modelo.
"""

from sklearn.metrics import classification_report

model = SVC()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(classification_report(y_test, pred))


# https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html#sphx-glr-auto-examples-model-selection-plot-grid-search-digits-py
from sklearn.model_selection import GridSearchCV

# definindo os hiperparâmetros e seus respectivos valores a serem testados
tuned_parameters = [
    {"kernel": ['rbf', 'sigmoid'],
     "gamma": [1e-1, 1e-2, 1e-3, "auto"],
     "C": [1, 10, 100]
     },
]
model = SVC()


search = GridSearchCV(model, tuned_parameters, scoring="accuracy", cv=5, refit=True)
result = search.fit(X_train, y_train)

# na variável result, você encontrará várias informações sobre a análise realizada.
dir(result)

# Uma delas é o cv_results_, onde você encontrará as informações sobre as análises do CV de cada modelo
result.cv_results_


# 10 melhores combinações de hiperparâmetros
p = pd.concat([pd.DataFrame(result.cv_results_["params"]),
               pd.Series(result.cv_results_["rank_test_score"], name="rank_test_score"),
               pd.Series(result.cv_results_["mean_test_score"], name="mean_test_score")], axis=1)
p[result.cv_results_["rank_test_score"] <= 10].sort_values("rank_test_score")

# o melhor modelo encontrado se encontra no atributo best_estimator_
best_model = result.best_estimator_

best_model

# avaliar o modelo com os dados de validação
yhat = best_model.predict(X_test)
acc = classification_report(y_test, yhat)
print(acc)


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])

# treinando e avaliando o modelo com os valores padrões
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
print(classification_report(y_test, pred))

# hiperparâmetros a serem ajustados
tuned_parameters2 = [
    {"svc__kernel": ['rbf', 'sigmoid'], "svc__gamma": [1e-1, 1e-2, 1e-3, "auto"], "svc__C": [1, 10, 100]},
]

search = GridSearchCV(pipe, tuned_parameters2, scoring="accuracy", cv=5, refit=True)
result = search.fit(X_train, y_train)

# 10 melhores combinações de hiperparâmetros
p = pd.concat([pd.DataFrame(result.cv_results_["params"]),
               pd.Series(result.cv_results_["rank_test_score"], name="rank_test_score"),
               pd.Series(result.cv_results_["mean_test_score"], name="mean_test_score")], axis=1)
p[result.cv_results_["rank_test_score"] <= 10].sort_values("rank_test_score")

best_model = result.best_estimator_

best_model

# avaliar o modelo com os dados de validação
yhat = best_model.predict(X_test)
acc = classification_report(y_test, yhat)
print(acc)

"""## HalvingGridSearchCV"""

from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV

gsh = HalvingGridSearchCV(estimator=pipe, param_grid=tuned_parameters2, scoring="accuracy", factor=2)
result = gsh.fit(X_train, y_train)

# 10 melhores combinações de hiperparâmetros
p = pd.concat([pd.DataFrame(result.cv_results_["params"]),
               pd.Series(result.cv_results_["rank_test_score"], name="rank_test_score"),
               pd.Series(result.cv_results_["mean_test_score"], name="mean_test_score")], axis=1)
p[result.cv_results_["rank_test_score"] <= 10].sort_values("rank_test_score")

best_model = result.best_estimator_

best_model

# avaliar o modelo com os dados de validação
yhat = best_model.predict(X_test)
acc = classification_report(y_test, yhat)
print(acc)

"""## RandomizedSearchCV"""

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

# specify parameters and distributions to sample from
param_dist = {
    "svc__kernel": ['rbf', 'sigmoid'],
    "svc__gamma": loguniform(1e-3, 1e-1),
    "svc__C": loguniform(1e0, 1e2),
}
# run randomized search
n_iter_search = 30
random_search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=n_iter_search)

result = random_search.fit(X_train, y_train)

# 10 melhores combinações de hiperparâmetros
p = pd.concat([pd.DataFrame(result.cv_results_["params"]),
               pd.Series(result.cv_results_["rank_test_score"], name="rank_test_score"),
               pd.Series(result.cv_results_["mean_test_score"], name="mean_test_score")], axis=1)
p[result.cv_results_["rank_test_score"] <= 10].sort_values("rank_test_score")

p

best_model = result.best_estimator_

best_model

# %%

# avaliar o modelo com os dados de validação
yhat = best_model.predict(X_test)
acc = classification_report(y_test, yhat)
print(acc)

# %%

"""### Exercício

Tente encontrar as melhores combinações de hiperparâmetros para o método da regressão logística utilizando os mesmos dados. Teste os seguintes valores dos seguintes hiperparâmetros:

*   penalty: 'l2', 'none';
*   C: [0.001, 0.01, 0.1, 1, 10, 100];
*   solver: 'lbfgs', 'newton-cg', 'newton-cholesky', 'sag';
*   max_iter: [100, 200, 500].


"""
# %%
from sklearn.linear_model import LogisticRegression

# definindo os hiperparâmetros para regressão logística
tuned_parameters_lr = [
    {
        "penalty": ['l2', 'none'],
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "solver": ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag'],
        "max_iter": [100, 200, 500]
    }
]

# criando o modelo de regressão logística
model_lr = LogisticRegression()

# realizando a busca em grade
search_lr = GridSearchCV(model_lr, tuned_parameters_lr, scoring="accuracy", cv=5, refit=True)
result_lr = search_lr.fit(X_train, y_train)

# 10 melhores combinações de hiperparâmetros
p_lr = pd.concat([pd.DataFrame(result_lr.cv_results_["params"]),
                  pd.Series(result_lr.cv_results_["rank_test_score"], name="rank_test_score"),
                  pd.Series(result_lr.cv_results_["mean_test_score"], name="mean_test_score")], axis=1)
print("Top 10 combinações:")
print(p_lr[result_lr.cv_results_["rank_test_score"] <= 10].sort_values("rank_test_score"))

# %% 
# melhor modelo encontrado
best_model_lr = result_lr.best_estimator_
print("\nMelhor modelo:")
print(best_model_lr)

# avaliar o modelo com os dados de validação
yhat_lr = best_model_lr.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, yhat_lr))

# %%

# usando Pipeline com StandardScaler para melhorar a convergência
pipe_lr = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression())])

# hiperparâmetros com prefixo 'lr__' para o pipeline
tuned_parameters_lr_pipe = [
    {
        "lr__penalty": ['l2', 'none'],
        "lr__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "lr__solver": ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag'],
        "lr__max_iter": [100, 200, 500]
    }
]

search_lr_pipe = GridSearchCV(pipe_lr, tuned_parameters_lr_pipe, scoring="accuracy", cv=5, refit=True)
result_lr_pipe = search_lr_pipe.fit(X_train, y_train)

# 10 melhores combinações
p_lr_pipe = pd.concat([pd.DataFrame(result_lr_pipe.cv_results_["params"]),
                       pd.Series(result_lr_pipe.cv_results_["rank_test_score"], name="rank_test_score"),
                       pd.Series(result_lr_pipe.cv_results_["mean_test_score"], name="mean_test_score")], axis=1)
print("Top 10 combinações com scaling:")
print(p_lr_pipe[result_lr_pipe.cv_results_["rank_test_score"] <= 10].sort_values("rank_test_score"))

best_model_lr_pipe = result_lr_pipe.best_estimator_
print("\nMelhor modelo com scaling:")
print(best_model_lr_pipe)

# avaliar o modelo
yhat_lr_pipe = best_model_lr_pipe.predict(X_test)
print("\nClassification Report com scaling:")
print(classification_report(y_test, yhat_lr_pipe))

# %%
