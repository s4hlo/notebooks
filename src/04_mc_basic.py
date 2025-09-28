# %%
import numpy as np
from typing import Tuple
from utils.ambiente import AmbienteNavegacaoLabirinto
from utils.plot import plot_policy, plot_tabular


# %%
def gerar_episodio(
    ambiente,
    estado_inicial: int,
    acao_inicial: int,
    Pi: np.ndarray,
    T: int,
    gamma: float,
) -> float:
    """
    Executa um episódio Monte Carlo de comprimento fixo T. O primeiro passo usa a ação forçada "acao_inicial", os demais seguem a política fornecida (Pi).

    Parâmetros
    ----------
    ambiente : AmbienteNavegacaoLabirinto
    estado_inicial : int
        Índice linear do estado inicial.
    acao_inicial : int
        Ação forçada no primeiro passo -> necessária para cobrir todo par (s,a).
    Pi : np.ndarray, shape (n_estados, n_acoes)
        Política seguida a partir do segundo passo.
    T : int
        Comprimento fixo do episódio.
    gamma : float
        Fator de desconto.

    Retorna
    -------
    G : float
        Retorno acumulado.
    """

    # Código aqui
    # Dica: Utilize os métodos reset_to_state e step do ambiente

    return G


# %%
def avaliar_politica_mc(
    ambiente,
    Pi: np.ndarray,
    gamma: float,
    T: int,
    N: int,
) -> np.ndarray:
    """
    Estima Q(s,a) por Monte Carlo com episódios de comprimento fixo T.
    Para cada par (s,a), gera N episódios e faz a média dos retornos.

    Parâmetros
    ----------
    ambiente : AmbienteNavegacaoLabirinto
        Instância do gridworld.
    Pi : np.ndarray, shape (n_estados, n_acoes)
        Política a ser avaliada.
    gamma : float
        Fator de desconto.
    T : int
        Horizonte fixo (número de passos por episódio).
    N : int
        Número de episódios gerados para cada par (s,a).
    """

    # Atalhos do ambiente (shapes)
    n_estados = ambiente.n_states  # int
    n_acoes = ambiente.n_actions  # int

    # Inicializações
    Q = np.zeros((n_estados, n_acoes), dtype=float)  # armazena as médias
    retornos = np.zeros((n_estados, n_acoes), dtype=float)  # acumula os retornos

    # Código aqui

    return Q


# %%
def melhorar_politica(
    ambiente,
    Q: np.ndarray,
) -> np.ndarray:
    """
    Gera a política gulosa determinística em relação aos valores de ação Q.

    Parâmetros
    ----------
    ambiente : AmbienteNavegacaoLabirinto
        Instância do gridworld.
    Q : np.ndarray, shape (n_estados, n_acoes)
        Valores-ação estimados.

    Retorna
    -------
    Pi_nova : np.ndarray, shape (n_estados, n_acoes)
        Política gulosa determinística.
    """

    # Atalhos do ambiente (shapes)
    n_estados = ambiente.n_states  # int
    n_acoes = ambiente.n_actions  # int

    # Código aqui

    return Pi_nova


# %%
def mc_basico(
    ambiente,
    gamma: float = 0.9,
    N: int = 1,  # nº de episódios por (s,a)
    max_iter: int = 50,  # nº de ciclos (avaliação + melhoria)
    T: int = 50,  # horizonte do episódio
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Monte Carlo Básico (variante sem modelo da iteração de política) no gridworld.

    O método alterna:
      1) Avaliação de política (MC): Q(s,a) = média de N retornos gerados com
         episódios de comprimento fixo T, usando ação inicial forçada a em s.
      2) Melhoria de política: Torna a política determinística e gulosa em relação às estimativas obtidas de Q.


    Parâmetros
    ----------
    ambiente : AmbienteNavegacaoLabirinto
        Instância do gridworld.
    gamma : float
        Fator de desconto.
    N : int
        Número de episódios por par (s,a) na avaliação de política.
    max_iter : int
        Número de ciclos (avaliação+melhoria) a executar.
    T : int
        Horizonte (número de passos) de cada episódio MC.

    Retorna
    -------
    Q  : np.ndarray, shape (n_estados, n_acoes)
        Estimativa de Q(s,a) ao final do último ciclo.
    Pi : np.ndarray, shape (n_estados, n_acoes)
        Política determinística resultante.
    k  : int
        Número de ciclos efetivamente executados (1..max_iter).
    """

    # Atalhos do ambiente (shapes)
    n_estados = ambiente.n_states  # int
    n_acoes = ambiente.n_actions  # int

    # Inicializações
    Pi = np.zeros((n_estados, n_acoes), dtype=float)  # Política inicial determinística:
    Pi[:, 0] = 1.0  # ação 0 em todos os estados
    Q = np.zeros((n_estados, n_acoes), dtype=float)

    for k in tqdm(range(1, max_iter + 1), desc="Iterações (MC Básico)"):
        # 1) Avaliação de política por MC (média de N episódios por (s,a))
        Q = avaliar_politica_mc(ambiente, Pi, gamma, T, N)

        # 2) Melhoria de política (gulosa determinística)
        Pi = melhorar_politica(ambiente, Q)

    return Q, Pi, k


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
Q, Pi, k = mc_basico(
    ambiente,  # gridworld
    gamma=0.9,  # fator de desconto
    N=1,  # número de episódios por (s,a)
    max_iter=20,  # número de ciclos (avaliação + melhoria) a executar
    T=100,  # horizonte fixo por episódio
)

# Derivar V a partir de Q:
V = np.sum(Pi * Q, axis=1)

# %% [markdown]
# ### Visualização

# %%
plot_tabular(Q, kind="Q")
plot_tabular(Pi, kind="Pi")
plot_tabular(V, kind="V", ambiente=ambiente, center_zero=False)
_ = plot_policy(ambiente, Pi)
