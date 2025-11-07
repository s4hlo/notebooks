# %% [markdown]
# # Laboratório 9B: n-step Sarsa (CliffWalking)

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
# ## Algoritmo: n-step Sarsa

# %%
# Alvo n passos
def _alvo_n_passos(S, A, R, tau: int, n: int, T_end: int, gamma: float, Q: np.ndarray) -> float:
    Gtau = 0.0
    upper_sum = int(min(tau + n, T_end))        # limite superior do somatório
    lower_sum = tau + 1                         # limite inferior do somatório
    for i in range(lower_sum, upper_sum + 1):
        Gtau += (gamma ** (i - tau - 1)) * R[i]
    if (tau + n) < T_end:
        Gtau += (gamma ** n) * Q[S[tau + n], A[tau + n]]
    return Gtau

# Atualização dos valores de ação (n-step SARSA)
def _atualiza_Q(Q, experiencia, tau, n, T_end, alpha, gamma):
    S, A, R = experiencia
    td_target = _alvo_n_passos(S, A, R, tau, n, T_end, gamma, Q)
    td_error  = Q[S[tau], A[tau]] - td_target
    Q[S[tau], A[tau]] -= alpha * td_error

# Atualização da política
def _atualiza_Pi(Pi: np.ndarray, Q: np.ndarray, s: int, eps: float) -> None:
    """
    Deixa Pi[s, :] ε-suave em torno de argmax_a Q[s, a].
    """
    n_acoes = Q.shape[1]
    a_star = int(np.argmax(Q[s]))
    Pi[s, :] = eps / n_acoes
    Pi[s, a_star] += 1.0 - eps

# %%
def n_step_sarsa(
    env: gym.Env,
    n: int = 3,
    gamma: float = 0.9,
    N: int = 500,
    T: int = 1_000,
    epsilon: float = 0.1,
    alpha: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, List[int], List[float]]:
    """
    n-step SARSA  com política ε-gulosa.

    O algoritmo mantém buffers de sequência (S, A, R) e atualiza Q(S_τ, A_τ) usando o retorno n-passos:
        G_τ = sum_{i=τ+1}^{min(τ+n, T_end)} γ^{i-τ-1} R_i  +  1_{τ+n < T_end} · γ^n · Q(S_{τ+n}, A_{τ+n})
    Para n = 1, recupera-se o SARSA(0).

    Parâmetros
    ----------
    env : gymnasium.Env
        Ambiente do gymnasium (ex.: 'CliffWalking-v1', 'FrozenLake-v1').
    n : int
        Horizonte do retorno n-passos (n ≥ 1). Para n=1, equivale ao SARSA(0).
    gamma : float
        Fator de desconto (0 ≤ γ ≤ 1).
    N : int
        Número de episódios de treinamento.
    T : int
        Limite máximo de passos por episódio.
    epsilon : float
        Parâmetro ε da política ε-gulosa (0 ≤ ε ≤ 1) utilizada on-policy.
    alpha : float
        Taxa de aprendizado (0 < α ≤ 1).
    seed : int | None
        Semente de aleatoriedade.

    Retorna
    -------
    Q : np.ndarray, shape (n_states, n_actions)
        Estimativas finais dos valores de ação Q(s, a).
    Pi : np.ndarray, shape (n_states, n_actions)
        Política ε-suave final (linhas somam ≈ 1).
    numero_de_visitas : np.ndarray, shape (n_states, n_actions)
        Contagem de visitas a cada par (s, a) durante o treinamento.
    k : int
        Número de episódios efetivamente executados (== N).
    episodio_T : list[int]
        Duração (passos) de cada episódio, considerando término natural ou truncação em T.
    episodio_G : list[float]
        Retorno não-descontado (soma de recompensas) obtido em cada episódio.
    """

    rng = np.random.default_rng(seed)

    # Listas de métricas
    episodio_T = []
    episodio_G = []

    # Atalhos
    n_states  = env.observation_space.n
    n_actions = env.action_space.n

    # Inicializações
    Q                 = np.zeros((n_states, n_actions), dtype=float)
    numero_de_visitas = np.zeros((n_states, n_actions), dtype=float)

    # Política inicial ε-suave: uniforme
    Pi = np.full((n_states, n_actions), 1.0 / n_actions, dtype=float)

    # Loop episódios
    for _ in tqdm(range(1, N + 1), desc=f"Episódios (n-step SARSA, n={n})", leave=True):

        # Reset do ambiente
        s, _ = env.reset()

        # Buffers para armazenar sequência de estados, ações e recompensas
        S = [s]  # estados: S[0] = s_0
        A = []   # ações: A[t] = a_t
        R = [0.0]  # recompensas: R[0] não usado, R[t] = r_t (recompensa após ação A[t-1])
        
        # Seleciona ação inicial
        probs = Pi[s]
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            probs = np.full(n_actions, 1.0 / n_actions)
        a = int(rng.choice(n_actions, p=probs))
        A.append(a)
        
        G = 0.0
        t = 0
        T_end = T
        
        while True:
            if t < T:
                # Executa ação A[t] e observa próximo estado e recompensa
                s_next, r, terminated, truncated, _ = env.step(a)
                
                S.append(s_next)  # S[t+1] = s_next
                R.append(r)       # R[t+1] = r
                G += r
                t += 1
                
                if terminated or truncated:
                    T_end = t
                
                if not (terminated or truncated) and t < T:
                    # Seleciona próxima ação usando política ε-gulosa
                    probs_next = Pi[s_next]
                    if probs_next.sum() > 0:
                        probs_next = probs_next / probs_next.sum()
                    else:
                        probs_next = np.full(n_actions, 1.0 / n_actions)
                    a_next = int(rng.choice(n_actions, p=probs_next))
                    A.append(a_next)
                    a = a_next
                    s = s_next
            
            # Atualiza Q para estados visitados há n passos (tau = t - n)
            tau = t - n
            if tau >= 0:
                s_tau = S[tau]
                a_tau = A[tau]
                numero_de_visitas[s_tau, a_tau] += 1
                
                # Calcula alvo n-passos
                td_target = _alvo_n_passos(S, A, R, tau, n, T_end, gamma, Q)
                
                # Atualização: Q(s_τ, a_τ) ← Q(s_τ, a_τ) + α[G_τ - Q(s_τ, a_τ)]
                Q[s_tau, a_tau] += alpha * (td_target - Q[s_tau, a_tau])
                
                # Atualiza política ε-gulosa
                _atualiza_Pi(Pi, Q, s_tau, epsilon)
            
            if terminated or truncated:
                # Atualiza estados restantes (tau de T_end - n até T_end - 1)
                for tau in range(max(0, T_end - n), T_end):
                    s_tau = S[tau]
                    a_tau = A[tau]
                    numero_de_visitas[s_tau, a_tau] += 1
                    
                    td_target = _alvo_n_passos(S, A, R, tau, n, T_end, gamma, Q)
                    Q[s_tau, a_tau] += alpha * (td_target - Q[s_tau, a_tau])
                    
                    _atualiza_Pi(Pi, Q, s_tau, epsilon)
                break
        
        episodio_T.append(t)
        episodio_G.append(G)

    return Q, Pi, numero_de_visitas, N, episodio_T, episodio_G

# %% [markdown]
# ## Experimento

# %% [markdown]
# ### Simulação

# %%
# Hiper-parâmetros principais
EPISODIOS = 5000  # @param {type:"integer"}  # número de episódios
NSTEP     = 1     # @param {type:"integer"}
ALPHA     = 0.01  # @param {type:"number"}
GAMMA     = 0.1   # @param {type:"number"}
EPSILON   = 0.1   # @param {type:"number"}
SEED      = 42    # @param {type:"integer"}

# %%
# Sarsa
Q, Pi, numero_de_visitas, k, T, G = n_step_sarsa(
    env,
    n=NSTEP,
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
estados, acoes, recompensas = simular_trajetoria_gym(Pi, ambiente, max_steps=200)
plot_trajetoria_gym(ambiente, estados, titulo="Trajetória (gulosa)")

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
# 1. Implemente o algoritmo n-step Sarsa para resolver o ambiente `'CliffWalking-v1'` do [gymnasium](https://gymnasium.farama.org/environments/toy_text/cliff_walking/).
# 2. Varie o hiperparametro (NSTEPS) e fixe os demais (ex.: ALPHA, GAMMA, EPSILON) (Obs>: 5 valores distintos).
#     - Para o estudo do hiperparâmetro plote:
#         - a duração do episódio por episódio
#         - a recompensa total por episodio
#         - Uma trajetória gulosa utilizando `plot_trajetoria_gym`.
#     - Observação: as curvas para cada estudo de hiperparâmetro devem estar na mesma figura.
# 3. Repita o procedimento para cada um dos hiperparametros.
# 4. Reporte suas observações.
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


