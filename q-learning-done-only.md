```python
# %%
# Valores base dos hiperparâmetros
BASE_EPISODIOS = 5000
BASE_ALPHA = 0.01
BASE_GAMMA = 0.9
BASE_EPSILON = 0.1
BASE_SEED = 42

# Valores a variar para cada hiperparâmetro (3 valores cada)
VALORES_EPISODIOS = [1000, 5000, 10000]
VALORES_ALPHA = [0.001, 0.01, 0.1]
VALORES_GAMMA = [0.7, 0.9, 0.99]
VALORES_EPSILON = [0.1, 0.3, 0.5]

```


```python
# %%
# Executa todos os experimentos
executar_experimento_hiperparametro(
    'EPISODIOS', VALORES_EPISODIOS,
    BASE_EPISODIOS, BASE_ALPHA, BASE_GAMMA, BASE_EPSILON, BASE_SEED
)

```

    ============================================================
    Experimento: Variando EPISODIOS
    ============================================================
    
    Executando com EPISODIOS=1000...



    Episódios (Q-learning):   0%|          | 0/1000 [00:00<?, ?it/s]


    
    Executando com EPISODIOS=5000...



    Episódios (Q-learning):   0%|          | 0/5000 [00:00<?, ?it/s]


    
    Executando com EPISODIOS=10000...



    Episódios (Q-learning):   0%|          | 0/10000 [00:00<?, ?it/s]



    
![png](q-learning-done-only_files/q-learning-done-only_1_6.png)
    


    
    Plotando trajetórias gulosas para cada valor de EPISODIOS...



    
![png](q-learning-done-only_files/q-learning-done-only_1_8.png)
    



    
![png](q-learning-done-only_files/q-learning-done-only_1_9.png)
    



    
![png](q-learning-done-only_files/q-learning-done-only_1_10.png)
    



```python

# %%
executar_experimento_hiperparametro(
    'ALPHA', VALORES_ALPHA,
    BASE_EPISODIOS, BASE_ALPHA, BASE_GAMMA, BASE_EPSILON, BASE_SEED
)

```

    ============================================================
    Experimento: Variando ALPHA
    ============================================================
    
    Executando com ALPHA=0.001...



    Episódios (Q-learning):   0%|          | 0/5000 [00:00<?, ?it/s]


    
    Executando com ALPHA=0.01...



    Episódios (Q-learning):   0%|          | 0/5000 [00:00<?, ?it/s]


    
    Executando com ALPHA=0.1...



    Episódios (Q-learning):   0%|          | 0/5000 [00:00<?, ?it/s]



    
![png](q-learning-done-only_files/q-learning-done-only_2_6.png)
    


    
    Plotando trajetórias gulosas para cada valor de ALPHA...



    
![png](q-learning-done-only_files/q-learning-done-only_2_8.png)
    



    
![png](q-learning-done-only_files/q-learning-done-only_2_9.png)
    



    
![png](q-learning-done-only_files/q-learning-done-only_2_10.png)
    



```python

# %%
executar_experimento_hiperparametro(
    'GAMMA', VALORES_GAMMA,
    BASE_EPISODIOS, BASE_ALPHA, BASE_GAMMA, BASE_EPSILON, BASE_SEED
)

```

    ============================================================
    Experimento: Variando GAMMA
    ============================================================
    
    Executando com GAMMA=0.7...



    Episódios (Q-learning):   0%|          | 0/5000 [00:00<?, ?it/s]


    
    Executando com GAMMA=0.9...



    Episódios (Q-learning):   0%|          | 0/5000 [00:00<?, ?it/s]


    
    Executando com GAMMA=0.99...



    Episódios (Q-learning):   0%|          | 0/5000 [00:00<?, ?it/s]



    
![png](q-learning-done-only_files/q-learning-done-only_3_6.png)
    


    
    Plotando trajetórias gulosas para cada valor de GAMMA...



    
![png](q-learning-done-only_files/q-learning-done-only_3_8.png)
    



    
![png](q-learning-done-only_files/q-learning-done-only_3_9.png)
    



    
![png](q-learning-done-only_files/q-learning-done-only_3_10.png)
    



```python

# %%
executar_experimento_hiperparametro(
    'EPSILON', VALORES_EPSILON,
    BASE_EPISODIOS, BASE_ALPHA, BASE_GAMMA, BASE_EPSILON, BASE_SEED
)


```

    ============================================================
    Experimento: Variando EPSILON
    ============================================================
    
    Executando com EPSILON=0.1...



    Episódios (Q-learning):   0%|          | 0/5000 [00:00<?, ?it/s]


    
    Executando com EPSILON=0.3...



    Episódios (Q-learning):   0%|          | 0/5000 [00:00<?, ?it/s]


    
    Executando com EPSILON=0.5...



    Episódios (Q-learning):   0%|          | 0/5000 [00:00<?, ?it/s]



    
![png](q-learning-done-only_files/q-learning-done-only_4_6.png)
    


    
    Plotando trajetórias gulosas para cada valor de EPSILON...



    
![png](q-learning-done-only_files/q-learning-done-only_4_8.png)
    



    
![png](q-learning-done-only_files/q-learning-done-only_4_9.png)
    



    
![png](q-learning-done-only_files/q-learning-done-only_4_10.png)
    



```python

```
