
# %%
import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, List, Union, Optional, Set
from tqdm.auto import tqdm
from IPython.display import Image

# Importa funções utilitárias comuns
from utils_rl import (
    plotar_metricas,
    visualizar_politica,
    plot_V_grid,
    plot_tabular,
    simular_trajetoria_gym,
    plot_trajetoria_gym,
    gerar_gif_simulacao
)

# %%
ambiente = 'CliffWalking-v1'
render_mode = 'rgb_array'  # retorna imagens do ambiente como arrays de pixels
env = gym.make(ambiente, render_mode=render_mode)

# %% [markdown]
# ## Algoritmo: Sarsa

# %%
def sarsa(
    env,
    gamma: float = 0.9,
    N: int = 500,               # episódios
    epsilon: float = 0.1,
    alpha: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, List[int], List[float]]:
    """
    SARSA(0) on-policy com política ε-gulosa (ε-suave) para ambientes Gymnasium de espaço discreto.

    Parâmetros
    ----------
    env : gymnasium.Env
        Ambiente com observation_space/action_space do tipo Discrete.
    gamma : float
        Fator de desconto.
    N : int
        Número de episódios.
    eps : float
        Parâmetro ε da política ε-gulosa (0 ≤ ε ≤ 1).
    alpha : float
        Taxa de aprendizado.
    seed : int | None
        Semente de aleatoriedade (usada para `np.random.default_rng` e opcionalmente em `env.reset`).

    Retorna
    -------
    Q   : np.ndarray, shape (n_states, n_actions)
        Estimativas finais Q(s,a).
    Pi  : np.ndarray, shape (n_states, n_actions)
        Política ε-suave resultante.
    numero_de_visitas : np.ndarray, shape (n_states, n_actions)
        Contagem de visitas por par (s,a).
    k   : int
        Número de episódios efetivamente executados (== N).
    episodio_T : list[int]
        Comprimento (passos) de cada episódio.
    episodio_G : list[float]
        Retorno (soma de recompensas não-descontadas) de cada episódio.
    """

    # Atalhos
    n_states  = env.observation_space.n
    n_actions = env.action_space.n

    rng = np.random.default_rng(seed)

    # Tabelas
    Q                 = np.zeros((n_states, n_actions), dtype=float)
    Pi                = np.full((n_states, n_actions), 1.0 / n_actions, dtype=float)
    numero_de_visitas = np.zeros((n_states, n_actions), dtype=float)

    episodio_T = []
    episodio_G = []

    for k in tqdm(range(1, N + 1), desc="Episódios (SARSA)", leave=True):
        # Reset para o estado inicial (s0)
        s, _ = env.reset()

        # Escolhe ação inicial usando política ε-gulosa (amostra de Pi[s])
        probs = Pi[s]
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            probs = np.full(n_actions, 1.0 / n_actions)
        a = int(rng.choice(n_actions, p=probs))

        # Inicializa métricas do episódio
        G = 0.0  # retorno (soma de recompensas não-descontadas)
        t = 0    # contador de passos

        ############################################################################
        # Implementação SARSA(0)
        while True:
            # Executa ação e observa transição
            s_next, r, terminated, truncated, _ = env.step(a)
            
            # Acumula recompensa e incrementa contador
            G += r
            t += 1
            
            # Incrementa contagem de visitas
            numero_de_visitas[s, a] += 1
            
            # Atualiza Q(s,a) usando SARSA
            if terminated or truncated:
                # Episódio terminou: Q(s',a') = 0
                Q[s, a] += alpha * (r - Q[s, a])
            else:
                # Escolhe próxima ação a' usando política ε-gulosa
                probs_next = Pi[s_next]
                if probs_next.sum() > 0:
                    probs_next = probs_next / probs_next.sum()
                else:
                    probs_next = np.full(n_actions, 1.0 / n_actions)
                a_next = int(rng.choice(n_actions, p=probs_next))
                
                # Atualização SARSA: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
                Q[s, a] += alpha * (r + gamma * Q[s_next, a_next] - Q[s, a])
                
                # Avança para próximo estado e ação
                s = s_next
                a = a_next
            
            # Atualiza política ε-suave para o estado s baseada em Q (sempre após atualizar Q)
            a_star = int(np.argmax(Q[s]))
            Pi[s, :] = epsilon / n_actions
            Pi[s, a_star] += 1.0 - epsilon
            
            # Verifica se episódio terminou
            if terminated or truncated:
                break
        
        # Registra métricas do episódio
        episodio_T.append(t)
        episodio_G.append(G)
        ############################################################################

    return Q, Pi, numero_de_visitas, N, episodio_T, episodio_G

# %% [markdown]
# ## Experimento

# %% [markdown]
# ### Simulação

# %%
# Hiper-parâmetros principais
EPISODIOS = 5000  # @param {type:"integer"}  # número de episódios
ALPHA     = 0.01  # @param {type:"number"}
GAMMA     = 0.9   # @param {type:"number"}
EPSILON   = 0.3   # @param {type:"number"}
SEED      = 42    # @param {type:"integer"}

# %%
# Sarsa
Q, Pi, numero_de_visitas, k, T, G = sarsa(
    env,
    gamma=GAMMA,
    N=EPISODIOS,
    epsilon=EPSILON,
    alpha=ALPHA,
    seed=SEED
)

# Derivar V a partir de Q:
V = np.sum(Pi * Q, axis=1)

# %% [markdown]
# ### Visualização

# %%
plotar_metricas(T, G)

# %%
visualizar_politica(Pi, ambiente)

# %%
estados, acoes, recompensas = simular_trajetoria_gym(Pi, "CliffWalking-v1", max_steps=200)
plot_trajetoria_gym("CliffWalking-v1", estados, titulo="Trajetória (gulosa)")

# %%
# Recria ambiente para renderizar
env = gym.make(ambiente, render_mode="rgb_array")
path_gif="politica.gif"
gif = gerar_gif_simulacao(Pi, env, path_gif=path_gif, n_episodios=10, greedy=False)

# %%
# Exibe o GIF diretamente no notebook
Image(filename=path_gif)

# %%
# Q: ndarray (n_estados, n_acoes)
plot_tabular(Q, kind="Q")

# %%
# Pi: ndarray (n_estados, n_acoes)
plot_tabular(Pi, kind="Pi")

# %%
# V: ndarray (n_estados,)
print(ambiente)
_ = plot_tabular(V, kind="V", env_name=ambiente, center_zero=False)

# %% [markdown]
# # Tarefa:
# 
# 1. Implemente o algoritmo Sarsa para resolver o ambiente `'CliffWalking-v1'` do [gymnasium](https://gymnasium.farama.org/environments/toy_text/cliff_walking/).
# 2. Considere os 4 hiperparametros (EPISODIOS, ALPHA, GAMMA, EPSILON)
#     - Varie um dos hiperparametros (ex.: EPISODIOS) e fixe os demais (ex.: ALPHA, GAMMA, EPSILON).
#         - Obs.: 3 valores para cada hiperparâmetro.
#     - Para cada estudo de hiperparâmetro plote:
#         - A duração do episódio por episódio
#         - A recompensa total por episodio
#         - Uma trajetória gulosa utilizando `plot_trajetoria_gym`.
#     - Observação: as curvas para cada estudo de hiperparâmetro devem estar na mesma figura, isto é, se o hiperparâmetro a ser variado é o EPSILON com 3 valores, então o gráfico de duração do episódio deve mostrar as 3 curvas relativas a cada valor de EPSILON (com legenda) e de maneira similar para a recompensa total por episodio.
# 3. Repita o procedimento para cada um dos hiperparametros.
# 4. Reporte suas observações.
# 
# **Entregáveis:**
# 
# 2. **Código**
# 1. **Relatório** (`*.pdf`).
# - O PDF deve conter:
#   - **Setup** (hiperparâmetros usados).
#   - **Resultados** (figuras e tabelas organizadas por experimento).
#   - **Análises curtas** por experimento.
# - O PDF **NÃO** deve conter:
#     - Códigos.


