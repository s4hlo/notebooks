# %% [markdown]
# # Laboratório 8B: Q-learning (CliffWalking)

# %% [markdown]
# ## Importações

# %%
# # Instala os pacotes necessários:
# # - gymnasium[toy-text]: inclui ambientes simples como FrozenLake, Taxi, etc.
# # - imageio[ffmpeg]: permite salvar vídeos e GIFs (formato .mp4 ou .gif)
!pip install gymnasium[toy-text] imageio[ffmpeg]

# %%
import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, List, Union, Optional, Set
from tqdm.auto import tqdm

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
# ## Algoritmo: Q-learning

# %%
# Desempenho da POLÍTICA ALVO (gulosa, determinística)
def avaliar_desempenho(
    env,
    Pi: np.ndarray,
    max_steps: int = 200,
) -> Tuple[int, float]:
    """
    Executa 1 episódio usando a política alvo (gulosa) e retorna (t, G) — passos e retorno não descontado.
    """
    s, _ = env.reset()
    G, t = 0.0, 0
    while t < max_steps:
        a = int(np.argmax(Pi[s]))
        s, r, terminated, truncated, _ = env.step(a)
        G += r
        t += 1
        if terminated or truncated:
            break
    return t, G

# Q-learning
def q_learning(
    env,
    gamma: float = 0.9,
    N: int = 500,
    epsilon: float = 0.1,
    alpha: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, List[int], List[float]]:
    """
    Q-learning com política de comportamento ε-gulosa (ε-suave) e política alvo gulosa.

    - A política de comportamento (ε-gulosa) é utilizada para explorar o ambiente.
    - A política alvo é determinística e gulosa (greedy) em relação aos valores de ação Q.
    - A atualização de Q utiliza o melhor valor de ação possível em s' (independente da ação executada pela política de comportamento).
    - O desempenho é avaliado após cada episódio executando a política alvo de forma determinística.

    Laço por episódio:
      1. Reinicia o ambiente: s0 ← env.reset().
      2. Seleciona a0 da política de comportamento: a0 ~ Pi_behavior(s0).
      3. Para cada passo:
         a. Executa a ação a_t no ambiente: observa (r_{t+1}, s_{t+1}, terminated, truncated).
         b. Atualiza Q(s_t, a_t):
            Q(s_t,a_t) ← Q(s_t,a_t) - α [Q(s_t,a_t) - (r_{t+1} + γ max_{a'} Q(s_{t+1},a'))].
         c. Atualiza a política alvo Pi_target(s_t) como gulosa em torno de argmax_a Q(s_t,a).
         d. Atualiza a política de comportamento Pi_behavior(s_t) para ser ε-suave em torno de argmax_a Q(s_t,a).
         e. Seleciona a_{t+1} da política de comportamento: a_{t+1} ~ Π_behavior(s_{t+1}).
         f. Avança o estado: s ← s_{t+1}, a ← a_{t+1}.
         g. Encerra o episódio se terminated ou truncated.
      4. Após cada episódio, executa a política alvo para medir o desempenho (retorno não descontado G e duração t).

    Parâmetros
    ----------
    env : gymnasium.Env
        Ambiente Gymnasium.
    gamma : float
        Fator de desconto (0 ≤ γ ≤ 1).
    N : int
        Número total de episódios.
    epsilon : float
        Parâmetro ε da política de comportamento ε-gulosa (0 ≤ ε ≤ 1).
    alpha : float
        Taxa de aprendizado (0 < α ≤ 1).
    seed : int | None
        Semente para reprodutibilidade dos sorteios aleatórios.

    Retorna
    -------
    Q : np.ndarray, shape (n_states, n_actions)
        Estimativas finais da função de ação Q(s,a).
    Pi_target : np.ndarray, shape (n_states, n_actions)
        Política alvo aprendida (determinística e gulosa em relação a Q).
    numero_de_visitas : np.ndarray, shape (n_states, n_actions)
        Contagem do número de visitas a cada par (s,a) durante o treinamento.
    N : int
        Número total de episódios efetivamente executados (igual ao parâmetro N).
    episodio_T : list[int]
        Lista contendo o número de passos de cada execução da política alvo após cada episódio.
    episodio_G : list[float]
        Lista contendo o retorno não descontado (soma das recompensas) da política alvo após cada episódio.
    """

    rng = np.random.default_rng(seed)

    # Métricas por episódio
    episodio_T = []
    episodio_G = []

    # Atalhos
    n_states  = env.observation_space.n
    n_actions = env.action_space.n

    # Inicializações
    Q                 = np.zeros((n_states, n_actions), dtype=float)                     # Valores de ação
    Pi_target         = np.zeros((n_states, n_actions), dtype=float)                     # Política alvo
    Pi_behavior       = np.full((n_states, n_actions), 1.0 / n_actions, dtype=float)     # Política de comportamento uniforme
    numero_de_visitas = np.zeros((n_states, n_actions), dtype=float)                     # Número de visitas

    for _ in tqdm(range(1, N + 1), desc="Episódios (Q-learning)", leave=True):
        s, _ = env.reset()

        ############################################################################
        # Implementação aqui
        # Dica:
        # usar
        # próximo estado, recompensa, terminated (flag), truncated (flag), _ = env.step(ação)
        # para fazer a trasição de um estado para o outro dada uma ação do agente




        ############################################################################

    return Q, Pi_target, numero_de_visitas, N, episodio_T, episodio_G

# %% [markdown]
# ## Experimento

# %% [markdown]
# ### Simulação

# %%
# Hiper-parâmetros principais
EPISODIOS = 5000  # @param {type:"integer"}  # número de episódios
ALPHA     = 0.01  # @param {type:"number"}
GAMMA     = 0.9   # @param {type:"number"}
EPSILON   = 0.1   # @param {type:"number"}
SEED      = 42    # @param {type:"integer"}

# %%
# Q-learning
Q, Pi, numero_de_visitas, k, T, G = q_learning(
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
# 1. Implemente o algoritmo Q-learning para resolver o ambiente `'CliffWalking-v1'` do [gymnasium](https://gymnasium.farama.org/environments/toy_text/cliff_walking/).
# 2. Considere os 4 hiperparametros (EPISODIOS, ALPHA, GAMMA, EPSILON)
#     - Varie um dos hiperparametros (ex.: EPISODIOS) e fixe os demais (ex.: ALPHA, GAMMA, EPSILON). Obs.: 3 valores para cada hiperparâmetro.
#     - Para cada estudo de hiperparâmetro plote:
#         - a duração do episódio por episódio
#         - a recompensa total por episodio
#         - Uma trajetória gulosa utilizando `plot_trajetoria_gym`.
#     - Observação: as curvas para cada estudo de hiperparâmetro devem estar na mesma figura, isto é, se o hiperparâmetro a ser variado é o EPSILON com 3 valores, entao o gráfico de duração do episódio deve mostrar as 4 curvas relativas a cada valor de EPSILON (com legenda) e de maneira similar para a recompensa total por episodio.
# 3. Repita o procedimento para cada um dos hiperparametros.
# 4. Reporte suas observações.
# 5. Compare com os resultados obtidos com o algoritmo SARSA.
# 
# **Observação:**
# - **Utilize os mesmos valores de hiperparâmetros do laboratório 7B (SARSA).**
# 
# **Entregáveis:**
# 
# 2. **Código**
# 1. **Relatório** (`*.pdf`).
# - O PDF deve conter:
#   - **Setup** (parâmetros usados).
#   - **Resultados** (figuras e tabelas organizadas por experimento).
#   - **Análises curtas** por experimento.
# - O PDF **NÃO** deve conter:
#     - Códigos.


