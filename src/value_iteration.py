# %%
import numpy as np
from typing import Tuple
from utils.ambiente import AmbienteNavegacaoLabirinto
from utils.plot import plot_policy, plot_tabular


# %%
def iteracao_de_valor(
    ambiente: "AmbienteNavegacaoLabirinto",
    gamma: float = 0.9,
    theta: float = 1e-6,
    max_iteracoes: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if not (0.0 <= gamma < 1.0):
        raise ValueError(
            "Sem estados terminais, use 0 <= gamma < 1 para garantir convergência."
        )

    n_estados = ambiente.n_states
    n_acoes = ambiente.n_actions
    n_recompensas = ambiente.n_rewards
    R = ambiente.recompensas_imediatas
    T = ambiente.transicao_de_estados
    Ps = ambiente.state_transition_probabilities
    Pr = ambiente.reward_probabilities
    r_vector = ambiente.recompensas_possiveis

    V = np.zeros(n_estados)
    Q = np.zeros((n_estados, n_acoes))
    Pi = np.zeros((n_estados, n_acoes))

    for k in range(max_iteracoes):
        V_old = V.copy()

        for s in range(n_estados):
            for a in range(n_acoes):
                R_sa = 0.0
                for r_idx in range(n_recompensas):
                    R_sa += Pr[r_idx, s, a] * r_vector[r_idx]
                future_value = 0.0
                for s_next in range(n_estados):
                    future_value += Ps[s_next, s, a] * V[s_next]
                Q[s, a] = R_sa + gamma * future_value

        for s in range(n_estados):
            best_action = np.argmax(Q[s, :])
            Pi[s, :] = 0.0
            Pi[s, best_action] = 1.0

        for s in range(n_estados):
            V[s] = np.max(Q[s, :])

        delta = np.max(np.abs(V - V_old))
        if delta < theta:
            break

    return V, Q, Pi, k + 1


# %% [markdown]
# ## Experimentos

# %%
ambiente = AmbienteNavegacaoLabirinto(
    world_size=(5, 5),
    bad_states=[(1, 1), (1, 2), (2, 2), (3, 1), (3, 3), (4, 1)],
    target_states=[(3, 2)],
    allow_bad_entry=True,
    rewards=[-1, -10, 1, 0],
)
ambiente.plot_labirinto()
# %%
V, Q, Pi, k = iteracao_de_valor(ambiente, gamma=0.9, theta=1e-6, max_iteracoes=1000)

# %% [markdown]
# ### Visualização

# %%
plot_tabular(Q, kind="Q")
plot_tabular(Pi, kind="Pi")
plot_tabular(V, kind="V", ambiente=ambiente, center_zero=False)
_ = plot_policy(ambiente, Pi)

# %%
