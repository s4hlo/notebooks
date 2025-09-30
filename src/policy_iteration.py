# %%
import numpy as np
from typing import Tuple
from utils.ambiente import AmbienteNavegacaoLabirinto
from utils.plot import plot_policy, plot_tabular


# %%
def iteracao_de_politica(
    ambiente: "AmbienteNavegacaoLabirinto",
    gamma: float = 0.9,
    theta_avaliacao: float = 1e-6,  # tolerância do loop interno (avaliar_politica)
    theta_politica: float = 1e-6,  # tolerância do loop externo (||V - V_old||_inf)
    max_iteracoes_politica: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Iteração de Política.

    Executa ciclos de:
    1) avaliação da política até ||V_pi_k^(j) − V_pi_k^(j-1)||_inf < theta_avaliacao;
    2) melhoria gulosa da política em relação a Q.
    Para quando ||V_pi_k − V_pi_{k-1}||_inf < theta_politica ou ao atingir max_iteracoes_politica.

    Parâmetros
    ----------
    ambiente : AmbienteNavegacaoLabirinto
        Ambiente tabular com P(r|s,a) e P(s'|s,a).
    gamma : float, padrão=0.9
        Fator de desconto (0 ≤ gamma < 1).
    theta_avaliacao : float, padrão=1e-6
        Tolerância do loop interno (||V_pi_k^(j) − V_pi_k^(j-1)||_inf).
    theta_politica : float, padrão=1e-6
        Tolerância do loop externo (||V_pi_k − V_pi_{k-1}||_inf).
    max_iteracoes_politica : int, padrão=1000
        Limite máximo de ciclos (avaliação + melhoria).

    Retorna
    -------
    V : np.ndarray, shape (n_estados,)
        Valores de estado.
    Q : np.ndarray, shape (n_estados, n_acoes)
        Valores de ação.
    Pi : np.ndarray, shape (n_estados, n_acoes)
        Política determinística.
    num_iter : int
        Número de ciclos externos executados.
    """

    # Atalhos (utilize as variáveis que forem necessárias)
    n_estados = ambiente.n_states
    n_acoes = ambiente.n_actions
    n_recompensas = ambiente.n_rewards
    R = ambiente.recompensas_imediatas  # r(s,a) -> r
    T = ambiente.transicao_de_estados  # T(s,a) -> s'
    Ps = (
        ambiente.state_transition_probabilities
    )  # shape (n_estados, n_estados, n_acoes)
    Pr = ambiente.reward_probabilities  # shape (n_recompensas,  n_estados, n_acoes)
    r_vector = ambiente.recompensas_possiveis  # shape (n_recompensas,)

    # TODO: inicialize V, Q, Pi

    for _ in range(max_iteracoes_politica):

        # TODO: avaliação da política atual (usar theta_avaliacao para teste de convergência dos valores de estado)
        # Deve calcular:
        # Valores de estado V (numpy array) para todos os estados (shape = (ambiente.n_states, )) da política atual

        # TODO: MELHORIA DE POLÍTICA
        # Deve calcular:
        # Valores de ação Q (numpy array) para todos os estados (shape = (ambiente.n_states, ambiente.n_actions))
        # Política melhorada Pi (numpy array)

        # TODO: teste de convergência para os valores de estado (usar theta_politica para teste de convergência)
        pass

    # TODO: retorne V, Q, Pi, k+1

    raise NotImplementedError


# %% [markdown]
# ## Experimento

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
V, Q, Pi, k = iteracao_de_politica(
    ambiente,  # gridworld
    gamma=0.9,  # fator de desconto (0 <= gamma < 1)
    theta_avaliacao=1e-6,  # convergência do loop interno
    theta_politica=1e-6,  # convergência do loop externo
)
print(k)

# %% [markdown]
# ### Visualização

# %%
plot_tabular(Q, kind="Q")
plot_tabular(Pi, kind="Pi")
plot_tabular(V, kind="V", ambiente=ambiente, center_zero=False)
_ = plot_policy(ambiente, Pi)
