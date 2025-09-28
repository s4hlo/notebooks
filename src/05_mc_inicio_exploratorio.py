# %%
import numpy as np
from typing import Tuple
from tqdm.std import tqdm
from utils.ambiente import AmbienteNavegacaoLabirinto
from utils.plot import plot_policy, plot_tabular, plot_visitas_log

# %%
def gerar_episodio(
    ambiente,
    estado_inicial: int,
    acao_inicial: int,
    Pi: np.ndarray,
    T: int,
) -> List[Tuple[int, int, float]]:
    """
    Gera um episódio de comprimento fixo T atendendo à condição
    de inícios exploratórios: o primeiro par (estado_inicial, acao_inicial)=(s0, a0) é imposto.
    A partir do segundo passo segue a política determinística Pi.
    Cada passo armazenado na trajetória contém a tupla (s_t, a_t, r_{t+1}).

    Parâmetros
    ----------
    ambiente : AmbienteNavegacaoLabirinto
        Instância do gridworld.
    estado_inicial : int
        Índice linear do estado em que o episódio começa (s0).
    acao_inicial : int
        Ação forçada a0 em s0.
    Pi : np.ndarray, shape (n_estados, n_acoes)
        Política determinística one-hot (argmax define a ação).
    T : int
        Horizonte fixo do episódio - número de passos do episódio (>=1).

    Retorna
    -------
    trajetoria : List[Tuple[int, int, float]]
        Lista de tamanho T com os triplos (s_t, a_t, r_{t+1}).
    """

    if T < 1:
        raise ValueError("T deve ser >= 1.")

    # Preparação: "Teleporta" o agente para o estado inicial.
    ambiente.reset_to_state(estado_inicial)

    # Inicialização
    trajetoria: List[Tuple[int, int, float]] = []

    # Passo 0: ação forçada (inícios exploratórios)

    # Código aqui

    # Demais passos: seguem a política Pi

    # Código aqui

    return trajetoria


# %%
def mc_inicios_exploratorios(
    ambiente,
    gamma: float = 0.9,
    N: int = 10_000,
    T: int = 50,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Monte Carlo com Inícios Exploratórios (ES) no gridworld.

    Loop principal (por episódio):
      1) Escolhe uniformemente (s0, a0)  [condição ES];
      2) Gera trajetória de comprimento fixo T começando com (s0, a0) e depois seguindo Pi;
      3) Varredura reversa acumulando o retorno G e atualizando Q(s,a) por média (todas as visitas);
      4) Melhoria de política incremental após cada atualização de Q(s_t, . ), isto é, executa a etapa de melhoria de política (determinística e gulosa).

    Parâmetros
    ----------
    ambiente : AmbienteNavegacaoLabirinto
        Instância do gridworld.
    gamma : float
        Fator de desconto.
    N : int
        Número total de episódios (ciclos) a executar.
    T : int
        Horizonte fixo (número de passos) de cada episódio.

    Retorna
    -------
    Q   : np.ndarray, shape (n_estados, n_acoes)
        Estimativas finais de Q(s,a) obtidas por média de retornos.
    Pi  : np.ndarray, shape (n_estados, n_acoes)
        Política determinística resultante.
    numero_de_visitas : np.ndarray, shape (n_estados, n_acoes)
        Matriz com com o número de visitas por par (s,a).
    k   : int
        Número de episódios efetivamente executados.
    """

    # Reprodutibilidade
    rng = np.random.default_rng(seed)

    # Atalhos do ambiente (shapes)
    n_estados     = ambiente.n_states   # int
    n_acoes       = ambiente.n_actions  # int

    # Inicializações
    Q = np.zeros((n_estados, n_acoes), dtype=float)
    numero_de_visitas = np.zeros((n_estados, n_acoes), dtype=float)      # contagem de visitas
    soma_dos_retornos = np.zeros((n_estados, n_acoes), dtype=float)      # soma de retornos (para média empírica)

    # Política inicial determinística (ex.: ação 0 em todos os estados)
    Pi = np.zeros((n_estados, n_acoes), dtype=float)
    Pi[np.arange(n_estados), rng.integers(n_acoes, size=n_estados)] = 1.0

    for k in tqdm(range(1, N + 1), desc="Episódios (MC ES)"):

        # 1) CONDIÇÃO DE INÍCIOS EXPLORATÓRIOS: Escolha uniforme de (s0, a0)

        # Código aqui

        # 2) Gera trajetória com início exploratório

        # Código aqui

        # 3) Varredura reversa:
        # acumula as somas dos retornos G
        # atualiza Q por média de visitas
        # melhoria de política (gulosa em relação a Q no estado s_t)

        # Código aqui

    return Q, Pi, numero_de_visitas, k

# %% [markdown]
# ## Experimento

# %%
ambiente = AmbienteNavegacaoLabirinto(
        world_size=(5, 5),
        bad_states=[(1, 1), (1, 2), (2, 2), (3, 1), (3, 3), (4, 1)],
        target_states=[(3, 2)],
        allow_bad_entry=True,
        rewards=[-1, -10, 1, 0]
    )
ambiente.plot_labirinto()

# %%
Q, Pi, numero_de_visitas, k = mc_inicios_exploratorios(
    ambiente,   # gridworld
    gamma=0.9,  # fator de desconto
    N=10_000,   # número total de episódios
    T=100,      # comprimento fixo de cada episódio
    seed=0      # semente para a reprodutibilidade
)

# Derivar V a partir de Q:
# V = np.max(Q, axis=1)
V = np.sum(Pi * Q, axis=1)

# %% [markdown]
# ### Visualização

# %%
plot_tabular(Q, kind="Q")
plot_tabular(Pi, kind="Pi")
plot_tabular(numero_de_visitas, kind="VISITAS")
plot_visitas_log(numero_de_visitas)
plot_tabular(V, kind="V", ambiente=ambiente, center_zero=False)
_ = plot_policy(ambiente, Pi)
