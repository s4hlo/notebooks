# %% [markdown]
# # Laboratório 7B: Sarsa (CliffWalking)

# %% [markdown]
# ## Importações

# %%
# Instala os pacotes necessários:
# - gymnasium[toy-text]: inclui ambientes simples como FrozenLake, Taxi, etc.
# - imageio[ffmpeg]: permite salvar vídeos e GIFs (formato .mp4 ou .gif)
!pip install gymnasium[toy-text] imageio[ffmpeg]

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

# %% [markdown]
# ## Ambiente: nova instância
# %%
# Configura o ambiente
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


        ############################################################################
        # Implementação aqui
        # Dica:
        # usar
        # próximo estado, recompensa, terminated (flag), truncated (flag), _ = env.step(ação)
        # para fazer a trasição de um estado para o outro dada uma ação do agente




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


