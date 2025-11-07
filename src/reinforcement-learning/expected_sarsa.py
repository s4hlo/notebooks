# %%
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
import gymnasium as gym
import numpy as np
from typing import Tuple, List, Optional
from tqdm.auto import tqdm
from IPython.display import Image

# %%
ambiente = 'CliffWalking-v1'
render_mode = 'rgb_array'
env = gym.make(ambiente, render_mode=render_mode)

# %%
def expected_sarsa(
    env,
    gamma: float = 0.9,
    N: int = 500,
    epsilon: float = 0.1,
    alpha: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, List[int], List[float]]:
    """
    Implementação do algoritmo Expected SARSA.
    
    Expected SARSA é similar ao SARSA, mas usa o valor esperado de Q(s',a')
    sobre a política atual, em vez de usar uma ação específica.
    """

    n_states  = env.observation_space.n
    n_actions = env.action_space.n

    rng = np.random.default_rng(seed)

    Q                 = np.zeros((n_states, n_actions), dtype=float)
    Pi                = np.full((n_states, n_actions), 1.0 / n_actions, dtype=float)
    numero_de_visitas = np.zeros((n_states, n_actions), dtype=float)

    episodio_T = []
    episodio_G = []

    for k in tqdm(range(1, N + 1), desc="Episódios (Expected SARSA)", leave=True):
        s, _ = env.reset()

        probs = Pi[s]
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            probs = np.full(n_actions, 1.0 / n_actions)
        a = int(rng.choice(n_actions, p=probs))

        G = 0.0
        t = 0

        while True:
            s_next, r, terminated, truncated, _ = env.step(a)
            
            G += r
            t += 1
            
            numero_de_visitas[s, a] += 1
            
            if terminated or truncated:
                Q[s, a] += alpha * (r - Q[s, a])
            else:
                # Calcula valor esperado: Σ_a' π(a'|s') Q(s',a')
                probs_next = Pi[s_next]
                if probs_next.sum() > 0:
                    probs_next = probs_next / probs_next.sum()
                else:
                    probs_next = np.full(n_actions, 1.0 / n_actions)
                
                expected_value = np.sum(probs_next * Q[s_next, :])
                
                # Atualização Expected SARSA: Q(s,a) ← Q(s,a) + α[r + γΣ_a'π(a'|s')Q(s',a') - Q(s,a)]
                Q[s, a] += alpha * (r + gamma * expected_value - Q[s, a])
                
                # Seleciona próxima ação usando política ε-gulosa
                a = int(rng.choice(n_actions, p=probs_next))
                s = s_next
            
            a_star = int(np.argmax(Q[s]))
            Pi[s, :] = epsilon / n_actions
            Pi[s, a_star] += 1.0 - epsilon
            
            if terminated or truncated:
                break
        
        episodio_T.append(t)
        episodio_G.append(G)

    return Q, Pi, numero_de_visitas, N, episodio_T, episodio_G

# %%
EPISODIOS = 5000
ALPHA     = 0.01
GAMMA     = 0.9
EPSILON   = 0.3
SEED      = 42

# %%
Q, Pi, numero_de_visitas, k, T, G = expected_sarsa(
    env,
    gamma=GAMMA,
    N=EPISODIOS,
    epsilon=EPSILON,
    alpha=ALPHA,
    seed=SEED
)

V = np.sum(Pi * Q, axis=1)

# %%
plotar_metricas(T, G)
visualizar_politica(Pi, ambiente)
estados, acoes, recompensas = simular_trajetoria_gym(Pi, "CliffWalking-v1", max_steps=200)
plot_trajetoria_gym("CliffWalking-v1", estados, titulo="Trajetória (gulosa)")

# %%
env = gym.make(ambiente, render_mode="rgb_array")
path_gif="politica.gif"
gif = gerar_gif_simulacao(Pi, env, path_gif=path_gif, n_episodios=10, greedy=False)

# %%
Image(filename=path_gif)

# %%
plot_tabular(Q, kind="Q")

# %%
plot_tabular(Pi, kind="Pi")

# %%
print(ambiente)
_ = plot_tabular(V, kind="V", env_name=ambiente, center_zero=False)
