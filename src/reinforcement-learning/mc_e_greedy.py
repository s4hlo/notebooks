
import numpy as np
from typing import Tuple
from tqdm.std import tqdm
from utils.ambiente import AmbienteNavegacaoLabirinto
from utils.plot import plot_policy, plot_tabular, plot_visitas_log

# %%
def gerar_episodio(
    ambiente,
    Pi: np.ndarray,
    T: int,
) -> List[Tuple[int, int, float]]:
    """
    Gera um episódio de comprimento fixo T seguindo uma política epsilon-suave.

    Cada passo armazenado na trajetória contém a tupla (s_t, a_t, r_{t+1}).

    Parâmetros
    ----------
    ambiente : AmbienteNavegacaoLabirinto
        Instância do gridworld.
    Pi : np.ndarray, shape (n_estados, n_acoes)
        Política estocástica atual: Pi[s, a] = probabilidade de escolher a em s.
    T : int
        Horizonte fixo do episódio (>=1).

    Retorna
    -------
    trajetoria : List[Tuple[int, int, float]]
        Lista de tamanho T com os triplos (s_t, a_t, r_{t+1}).
    """
    if T < 1:
        raise ValueError("T deve ser >= 1.")
    
    # Atalhos do ambiente (shapes)
    n_estados = ambiente.n_states   # int
    n_acoes   = ambiente.n_actions  # int

    # Estado inicial sorteado uniformemente
    s_t = np.random.randint(n_estados)

    # Preparação: "Teleporta" o agente para o estado inicial.
    ambiente.reset_to_state(s_t)

    # Inicialização
    trajetoria: List[Tuple[int, int, float]] = []

    # Loop principal: segue a política estocástica epsilon-suave
    for _ in range(T):       

        # Código aqui

    return trajetoria


# %%
def mc_epsilon_guloso(
    ambiente,
    gamma: float = 0.9,
    N: int = 20,
    T: int = 100_000,
    eps: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Monte Carlo epsilon-guloso no gridworld.

    Loop por episódio:
      1) Gera trajetória de comprimento fixo T seguindo a política epsilon-suave Pi;
      2) Varredura reversa acumulando os retornos G e atualizando a estimativa de Q(s,a) por média (todas as visitas);
      3) Melhoria de política incremental: após cada atualização de Q(s_t, a_t), executa melhoria de política epsilon-gulosa em torno de argmax_a Q(s_t,a).

        
    Parâmetros
    ----------
    ambiente : AmbienteNavegacaoLabirinto
        Instância do gridworld.
    gamma : float
        Fator de desconto.
    N : int
        Número total de episódios.
    T : int
        Horizonte fixo (número de passos por episódio).
    eps : float
        Parâmetro epsilon da política epsilon-suave (0 <= epsilon <= 1).

    Retorna
    -------
    Q   : np.ndarray, shape (n_estados, n_acoes)
        Estimativas finais de Q(s,a).
    Pi  : np.ndarray, shape (n_estados, n_acoes)
        Política epsilon-suave resultante.
    numero_de_visitas : np.ndarray, shape (n_estados, n_acoes)
        Contagem de visitas por par (s,a).
    k   : int
        Número de episódios efetivamente executados (== N).
    """
   
    # Atalhos
    n_estados = ambiente.n_states
    n_acoes   = ambiente.n_actions

    # Inicializações
    Q                 = np.zeros((n_estados, n_acoes), dtype=float)
    soma_dos_retornos = np.zeros((n_estados, n_acoes), dtype=float)
    numero_de_visitas = np.zeros((n_estados, n_acoes), dtype=float)

    # Política inicial ε-suave: uniforme (equivale a ε-suave sem preferência inicial)
    Pi = np.full((n_estados, n_acoes), 1.0 / n_acoes, dtype=float)

    for k in tqdm(range(1, N + 1), desc="Episódios (MC epsilon-guloso)", leave=False):
        # 1. Gera episódio sob Pi (epsilon-suave)
        
        # Código aqui

        # 2. Varredura reversa: atualiza retornos (G's) e estimativas dos valores de ação (Q's)
                  
        # Código aqui

            # 3. Melhoria de política epsilon-suave incremental no estado s_t (dentro do loop do item 2. acima)
            
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
        rewards=[-1, -1, 1, 0]
    )
ambiente.plot_labirinto()

# %%
Q, Pi, numero_de_visitas, k = mc_epsilon_guloso(
    ambiente,   # gridworld
    gamma=0.9,  # fator de desconto
    N=20,   # número total de episódios
    T=100_000,      # comprimento fixo de cada episódio
    eps=0.5  # parâmetro epsilon da política epsilon-gulosa
)

# Derivar V a partir de Q:
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
