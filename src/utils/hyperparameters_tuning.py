# %%
# Importando as bibliotecas necessárias
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from scipy.stats import loguniform

# %%
# Preparação dos dados
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
data = pd.read_csv(url, delimiter=",", header=None)
header = ["age", "sex", "cp", "trestpbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
data.columns = header

# Selecionando features e target
X = data.iloc[:,0:13]
y = data.iloc[:,13]

# Transformando target em binário (valores > 0 = 1, senão = 0)
y = [1 if x > 0 else 0 for x in y]

# Removendo linhas com valores faltantes
id2drop = X[X.values == '?'].index
X = X.drop(id2drop)
y = pd.Series(y).drop(id2drop)

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
model_baseline = SVC()
model_baseline.fit(X_train, y_train)
pred_baseline = model_baseline.predict(X_test)
classification_report(y_test, pred_baseline)

# %%
# GridSearchCV
tuned_parameters = [
    {
        "kernel": ['rbf', 'sigmoid'],
        "gamma": [1e-1, 1e-2, 1e-3, "auto"],
        "C": [1, 10, 100]
    }
]

model = SVC()
search = GridSearchCV(model, tuned_parameters, scoring="accuracy", cv=5, refit=True)
result = search.fit(X_train, y_train)

results_df = pd.concat([
    pd.DataFrame(result.cv_results_["params"]),
    pd.Series(result.cv_results_["rank_test_score"], name="rank_test_score"),
    pd.Series(result.cv_results_["mean_test_score"], name="mean_test_score")
], axis=1)

results_df[result.cv_results_["rank_test_score"] <= 10].sort_values("rank_test_score")

best_model = result.best_estimator_
yhat = best_model.predict(X_test)
classification_report(y_test, yhat)


# %%
# Pipeline com StandardScaler
pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])

pipe.fit(X_train, y_train)
pred_pipe_baseline = pipe.predict(X_test)
classification_report(y_test, pred_pipe_baseline)
tuned_parameters_pipe = [
    {
        "svc__kernel": ['rbf', 'sigmoid'], 
        "svc__gamma": [1e-1, 1e-2, 1e-3, "auto"], 
        "svc__C": [1, 10, 100]
    }
]

search_pipe = GridSearchCV(pipe, tuned_parameters_pipe, scoring="accuracy", cv=5, refit=True)
result_pipe = search_pipe.fit(X_train, y_train)

results_pipe_df = pd.concat([
    pd.DataFrame(result_pipe.cv_results_["params"]),
    pd.Series(result_pipe.cv_results_["rank_test_score"], name="rank_test_score"),
    pd.Series(result_pipe.cv_results_["mean_test_score"], name="mean_test_score")
], axis=1)

results_pipe_df[result_pipe.cv_results_["rank_test_score"] <= 10].sort_values("rank_test_score")

best_model_pipe = result_pipe.best_estimator_
yhat_pipe = best_model_pipe.predict(X_test)
classification_report(y_test, yhat_pipe)

# %%
# HalvingGridSearchCV
gsh = HalvingGridSearchCV(estimator=pipe, param_grid=tuned_parameters_pipe, scoring="accuracy", factor=2)
result_halving = gsh.fit(X_train, y_train)

results_halving_df = pd.concat([
    pd.DataFrame(result_halving.cv_results_["params"]),
    pd.Series(result_halving.cv_results_["rank_test_score"], name="rank_test_score"),
    pd.Series(result_halving.cv_results_["mean_test_score"], name="mean_test_score")
], axis=1)

results_halving_df[result_halving.cv_results_["rank_test_score"] <= 10].sort_values("rank_test_score")

best_model_halving = result_halving.best_estimator_
yhat_halving = best_model_halving.predict(X_test)
classification_report(y_test, yhat_halving)

# %%
# RandomizedSearchCV
param_dist = {
    "svc__kernel": ['rbf', 'sigmoid'],
    "svc__gamma": loguniform(1e-3, 1e-1),
    "svc__C": loguniform(1e0, 1e2),
}

n_iter_search = 30
random_search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=n_iter_search, scoring="accuracy", cv=5, refit=True)
result_random = random_search.fit(X_train, y_train)

results_random_df = pd.concat([
    pd.DataFrame(result_random.cv_results_["params"]),
    pd.Series(result_random.cv_results_["rank_test_score"], name="rank_test_score"),
    pd.Series(result_random.cv_results_["mean_test_score"], name="mean_test_score")
], axis=1)

results_random_df[result_random.cv_results_["rank_test_score"] <= 10].sort_values("rank_test_score")

best_model_random = result_random.best_estimator_
yhat_random = best_model_random.predict(X_test)
classification_report(y_test, yhat_random)

# %%
# Regressão Logística
tuned_parameters_lr = [
    {
        "penalty": ['l2', 'none'],
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "solver": ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag'],
        "max_iter": [100, 200, 500]
    }
]

model_lr = LogisticRegression()
search_lr = GridSearchCV(model_lr, tuned_parameters_lr, scoring="accuracy", cv=5, refit=True)
result_lr = search_lr.fit(X_train, y_train)

results_lr_df = pd.concat([
    pd.DataFrame(result_lr.cv_results_["params"]),
    pd.Series(result_lr.cv_results_["rank_test_score"], name="rank_test_score"),
    pd.Series(result_lr.cv_results_["mean_test_score"], name="mean_test_score")
], axis=1)

results_lr_df[result_lr.cv_results_["rank_test_score"] <= 10].sort_values("rank_test_score")

best_model_lr = result_lr.best_estimator_
yhat_lr = best_model_lr.predict(X_test)
classification_report(y_test, yhat_lr)

# %%
# Pipeline Regressão Logística
pipe_lr = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression())])

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

results_lr_pipe_df = pd.concat([
    pd.DataFrame(result_lr_pipe.cv_results_["params"]),
    pd.Series(result_lr_pipe.cv_results_["rank_test_score"], name="rank_test_score"),
    pd.Series(result_lr_pipe.cv_results_["mean_test_score"], name="mean_test_score")
], axis=1)

results_lr_pipe_df[result_lr_pipe.cv_results_["rank_test_score"] <= 10].sort_values("rank_test_score")

best_model_lr_pipe = result_lr_pipe.best_estimator_
yhat_lr_pipe = best_model_lr_pipe.predict(X_test)
classification_report(y_test, yhat_lr_pipe)
