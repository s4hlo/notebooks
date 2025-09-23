# %% [markdown]
# # Laboratório 3: Iteração de política truncada

# %% [markdown]
# ## Importações

# %%
# In_estadostala os pacotes necessários:
# - gymnasium[toy-text]: inclui ambientes simples como FrozenLake, Taxi, etc.
# - imageio[ffmpeg]: permite salvar vídeos e GIFs (formato .mp4 ou .gif)
!pip install gymnasium[toy-text] imageio[ffmpeg]

# %%
# Importa as bibliotecas principais
import gymnasium as gym               # Biblioteca de simulações de ambientes para RL
import imageio                        # Usada para salvar a sequência de frames como GIF
from IPython.display import Image     # Para exibir a imagem (GIF) diretamente no notebook
import numpy as np                    # Importa o pacote NumPy, amplamente utilizado para manipulação de arrays e operações numéricas
from numpy import linalg as LA        # Rotinas de álgebra linear do NumPy (ex.: normas, autovalores, decomposições)
import matplotlib.pyplot as plt       # Biblioteca para criação de gráficos estáticos em Python (parte do matplotlib)
import seaborn as sns                 # Biblioteca baseada em matplotlib para gráficos estatísticos com visualização mais bonita (usada aqui para heatmaps)
from typing import Dict, Tuple, Optional, List  # Importa ferramentas de tipagem estática do Python

# %% [markdown]
# ## Ambiente: nova instância  do FrozenLake

# %%
map_name = '4x4'        # tamanho do mapa (pode ser '4x4' ou '8x8')
render_mode="rgb_array" # render_mode="rgb_array": retorna imagen_estados do ambiente como arrays de pixels
is_slippery=False       # is_slippery=True: torna o ambiente estocástico (com escorregões)

env = gym.make("FrozenLake-v1",
               map_name=map_name,
               render_mode=render_mode,
               is_slippery=is_slippery)
env = env.unwrapped     # ESSENCIAL para acessar env.P

################################################################################
# Estrutura de env.P
################################################################################
# env.P: Dict[int, Dict[int, List[Tuple[float, int, float, bool]]]]
# env.P = {
#     estado_0: {
#         acao_0: [(p, s', r, done), ...],
#         acao_1: [(p, s', r, done), ...],
#         ...
#     },
#     estado_1: {
#         acao_0: [(p, s', r, done), ...],
#         ...
#     },
#     ...
# }
# (p, s', r, done) = (probabilidade, proximo_estado, recompensa, finalizado)
# probabilidade = p(s',r|s,a)
################################################################################

# %% [markdown]
# ## Funções auxiliares para visualização

# %%
def visualizar_politica(
    Pi: np.ndarray,
    env,
    *,
    action_labels: Optional[List[str]] = None,   # default FrozenLake: ["←","↓","→","↑"]
    destacar_gulosa: bool = True,
    suptitle: Optional[str] = "Política (distribuições por estado)",
) -> None:
    # Inferir grid do Gymnasium (FrozenLake)
    if not (hasattr(env, "unwrapped") and hasattr(env.unwrapped, "desc")):
        raise ValueError("Passe um ambiente Gymnasium com 'env.unwrapped.desc' (ex.: FrozenLake-v1).")
    desc = env.unwrapped.desc
    desc_str = np.char.decode(desc, "utf-8") if getattr(desc.dtype, "kind", "") == "S" else desc.astype(str)

    n_rows, n_cols = desc_str.shape
    n_estados, n_acoes = Pi.shape
    if n_rows * n_cols != n_estados:
        raise ValueError(f"Incompatibilidade: grid {n_rows}x{n_cols} != n_estados={n_estados}.")

    # Máscaras de terminais
    holes = (desc_str == "H")
    goal  = (desc_str == "G")

    # Rótulos das ações (FrozenLake: LEFT, DOWN, RIGHT, UP)
    if action_labels is None:
        action_labels = ["←", "↓", "→", "↑"]
    if len(action_labels) != n_acoes:
        action_labels = [f"a{i}" for i in range(n_acoes)]

    # Figura
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.2 * n_rows))
    axs = np.array(axs).reshape(-1)

    for s in range(n_estados):
        r, c = divmod(s, n_cols)
        ax = axs[s]

        # Estados terminais: só fundo colorido, sem barras
        if holes[r, c] or goal[r, c]:
            if holes[r, c]:
                ax.set_facecolor((1.0, 0.0, 0.0, 0.15))  # vermelho translúcido
                ax.set_title(f"Estado {s} (H)")
            else:
                ax.set_facecolor((0.0, 1.0, 0.0, 0.15))  # verde translúcido
                ax.set_title(f"Estado {s} (G)")
            # esconder eixos e ticks
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
            continue

        # Estados não-terminais: barras
        pi = Pi[s].astype(float)
        tot = pi.sum()
        if tot > 0:
            pi /= tot  # normalização defensiva

        acoes = np.arange(n_acoes)
        colors = ["gray"] * n_acoes
        if destacar_gulosa:
            colors[int(np.argmax(pi))] = "dimgray"

        ax.bar(acoes, pi, color=colors)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(acoes)
        ax.set_xticklabels(action_labels)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_title(f"Estado {s}")

    if suptitle:
        fig.suptitle(suptitle, y=1.02, fontsize=12)

    plt.tight_layout()
    plt.show()

def plot_tabular(
    data,
    kind: str = "Q",          # "Q" (valores de ação), "Pi" (política), "V" (valores de estado)
    ambiente=None,            # necessário quando kind="V" para reshape
    ax=None,
    cbar: bool = True,
    fmt: str = ".1f",
    center_zero: bool = True  # só relevante para "Q" e "V"
):
    """
    Plota matrizes tabulares de RL em formato de heatmaps (mapas de calor).
    Esta função cobre 3 casos:
    1. kind="Q": heatmap de Q(s, a) com ações nas linhas e estados nas colunas.
    2. kind="Pi": heatmap de Pi(a|s) (probabilidades) com ações nas linhas e estados nas colunas.
    3. kind="V": heatmap de V(s) no grid (n_rows x n_cols) do ambiente .

    Parameters
    ----------
    data : ndarray
        Dados a serem plotados.
        - Para kind="Q" ou "Pi": array 2D com shape (n_estados, n_acoes).
        - Para kind="V": array 1D com shape (n_estados,) que será remodelado para (ambiente.n_rows, ambiente.n_cols).
    kind : {"Q", "Pi", "V"}, default="Q"
        Tipo do plot:
        - "Q" usa paleta divergente centrada em zero.
        - "Pi" usa paleta sequencial no intervalo [0, 1].
        - "V" plota o valor de estado no grid do ambiente.
    ambiente : object, optional
        Necessário quando kind="V". Deve expor n_rows e n_cols para o reshape.
    ax : matplotlib.axes.Axes, optional
        Eixo onde o heatmap será desenhado. Se None, uma nova figura/eixo é criado.
    cbar : bool, default=True
        Se True, exibe a barra de cores (colorbar).
    fmt : str, default=".1f"
        Formatação dos valores anotados em cada célula do heatmap.
    center_zero : bool, default=True
        Quando kind é "Q" ou "V", centraliza a escala de cores em zero (vmin=-absmax, vmax=absmax). Ignorado para "Pi".

    Returns
    -------
    ax : matplotlib.axes.Axes
        Eixo contendo o heatmap resultante.
    """
    kind = kind.upper()

    xlabel = {"V": "Colunas", "PI": "Estados", "Q": "Estados"}
    ylabel = {"V": "Linhas", "PI": "Ações", "Q": "Ações" }
    title  = {"V": "Valores de Estado (V(s))", "PI": r"Política ($\pi(a|s)$ transposta)", "Q": "Valores de ação (Q(s, a) transposta)"}

    fig = None

    #  V(s): precisa do shape do grid
    match kind:
        case "V":

            if ambiente is None:
                raise ValueError("Para kind='V', passe 'ambiente' para reshape (n_rows, n_cols).")

            if hasattr(ambiente, "n_rows") and hasattr(ambiente, "n_cols"):
                n_rows, n_cols = ambiente.n_rows, ambiente.n_cols
            elif hasattr(ambiente, "unwrapped") and hasattr(ambiente.unwrapped, "desc"):
                # ex.: FrozenLake-v1 (Gymnasium)
                n_rows, n_cols = ambiente.unwrapped.desc.shape
            else:
                raise ValueError(
                    "Passe um objeto com n_rows/n_cols ou um env Gymnasium com .unwrapped.desc."
                )

            M = data.reshape(n_rows, n_cols)

            if ax is None:
                fig, ax = plt.subplots(figsize=(n_cols, n_rows))

            if center_zero:
                vmax = float(np.abs(M).max())
                vmin = -vmax
            else:
                vmin = float(M.min())
                vmax = float(M.max())

            cmap, square = "bwr", True

        case "PI" | "Q":

            # Q(s,a) e Pi(a|s): ações nas linhas, estados nas colunas
            M = data.T  # data: (n_estados, n_acoes) -> transposto para (n_acoes, n_estados)
            n_acoes, n_estados = M.shape

            if ax is None:
                fig, ax = plt.subplots(figsize=(n_estados, n_acoes))

            if kind == "PI":
                cmap = "Blues";
                vmin, vmax = 0.0, 1.0
            else:  # "Q"
                cmap = "bwr"
                if center_zero:
                    vmax = float(np.abs(M).max())
                    vmin = -vmax
                else:
                    vmin = float(M.min())
                    vmax = float(M.max())

            square = False

        case _:
            raise ValueError(f"kind desconhecido: {kind!r} (use 'Q', 'Pi' ou 'V').")


    ax = sns.heatmap(
        data=M,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar=cbar,
        square=square,
        linewidths=0.5,
        linecolor="gray",
        ax=ax
    )

    ax.set_xlabel(xlabel[kind])
    ax.set_ylabel(ylabel[kind])
    ax.set_title(title[kind])

    # bordas externas
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.5)
        ax.spines[side].set_edgecolor("gray")

    # rótulos
    if kind in ("Q", "PI"):
        ax.set_xticks(np.arange(n_estados) + 0.5)
        ax.set_xticklabels([f"s{i}" for i in range(n_estados)], rotation=0)
        ax.set_yticks(np.arange(n_acoes) + 0.5)
        ax.set_yticklabels([f"a{i}" for i in range(n_acoes)], rotation=0)

    if fig is not None:
        plt.tight_layout()
        plt.show()

    return

# %% [markdown]
# ## Algoritmo: Iteração de política

# %% [markdown]
# ### Avaliar política

# %%
def avaliar_politica_truncada(
    env,
    Pi: np.ndarray,
    j_truncado: int,
    V: np.ndarray | None = None,
    gamma: float = 0.9,
) -> np.ndarray:
    """
    Avaliação de política truncada.

    Executa exatamente j_truncado varreduras do algoritmo de avaliação de política (Jacobi).

    Atualização:
        V_{new}(s) = sum_a Pi[a|s] * sum_{(p,s',r,done) in env.P[s][a]} p * [ r + gamma * V_old[s'] ]

    Parâmetros
    ----------
    env : gym.Env (unwrapped)
        Ambiente com dicionário de transições env.P: Dict[s][a] -> List[(p, s', r, done)].
    Pi : np.ndarray, shape (n_estados, n_acoes)
        Política.
    j_truncado : int
        Número de iterações (varreduras de avaliação).
    V : np.ndarray | None, shape (n_estados,), opcional
        Valores de estado iniciais. Se None, começa em zeros.
    gamma : float, default=0.9
        Fator de desconto.

    Retorna
    -------
    V : np.ndarray, shape (n_estados,)
        Valores de estado após j_truncado varreduras.
    """

    n_estados = env.observation_space.n
    n_acoes   = env.action_space.n

    # Inicializações
    if V is None:
        V = np.zeros(n_estados, dtype=float)

    ############################################################################################################
    # AVALIAÇÃO DA POLÍTICA ATUAL
    ############################################################################################################
    # Código aqui

    ############################################################################################################

    return V


# %% [markdown]
# ### Melhorar política

# %%
def melhorar_politica(
    env,
    V: np.ndarray,
    gamma: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Melhoria de política.

    Dado V, calcula:
        Q(s,a) = sum_{(p,s',r,done) in P[s][a]} p * (r + gamma * V[s'])
    e retorna a política gulosa determinística.

    Parâmetros
    ----------
    env : gym.Env (unwrapped)
        Ambiente com dicionário de transições env.P: Dict[s][a] -> List[(p, s', r, done)].
    V : np.ndarray, shape (n_estados,)
        Valores de estado.
    gamma : float, default=0.99
        Fator de desconto (0 <= gamma < 1).

    Retorna
    -------
    Q : np.ndarray, shape (n_estados, n_acoes)
        Valores de ação.
    Pi_new : np.ndarray, shape (n_estados, n_acoes)
        Política gulosa determinística.
    """
    # Dimensões e validações

    n_estados = env.observation_space.n
    n_acoes   = env.action_space.n

    # Inicializações
    Q      = np.zeros((n_estados, n_acoes), dtype=float)
    Pi_new = np.zeros((n_estados, n_acoes), dtype=float)

    ############################################################################################################
    # MELHORIA DA POLÍTICA
    ############################################################################################################
    # Código aqui

    ############################################################################################################

    return Q, Pi_new

# %% [markdown]
# ### Algoritmo: iteração de política truncada

# %%
def iteracao_de_politica_truncada(
    env,
    gamma: float = 0.99,
    j_truncado: int = 10,
    theta: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Iteração de Política truncada.

    Loop externo (k):
      1) V <- avaliar_politica_truncada(env, Pi_k, j_truncado, V, gamma)
      2) (Q, Pi_{k+1}) <- melhorar_politica(env, V, gamma)
      3) parar quando ||V - V_prev||_inf < theta

    Parâmetros
    ----------
    env : gym.Env (unwrapped)
        Ambiente com dicionário de transições env.P: Dict[s][a] -> List[(p, s', r, done)].
    gamma : float, default=0.9
        Fator de desconto (0 <= gamma < 1).
    j_truncado : int, default=10
        Número de varreduras na avaliação de política truncada.
    theta : float, default=1e-8
        Critério de parada baseado em V entre iterações externas (convergência).

    Retorna
    -------
    V  : np.ndarray, shape (n_estados,)
        Valores de estado.
    Q  : np.ndarray, shape (n_estados, n_acoes)
        Valores de ação da última melhoria.
    Pi : np.ndarray, shape (n_estados, n_acoes)
        Política determinística.
    k  : int
        Número de iterações externas executadas.
    """
    # Dimensões e validações básicas
    n_estados = env.observation_space.n
    n_acoes   = env.action_space.n

    # Inicializações
    V  = np.zeros(n_estados, dtype=float)
    Q  = np.zeros((n_estados, n_acoes), dtype=float)
    Pi = np.full((n_estados, n_acoes), 1.0 / n_acoes, dtype=float)  # política uniforme

    # Laço externo
    k = 0
    while True:

        k += 1

        V_prev = V.copy()

        # 1) Avaliação de política (truncada)
        V = avaliar_politica_truncada(
            env=env,
            Pi=Pi,
            j_truncado=j_truncado,
            V=V,
            gamma=gamma,
        )

        # 2) Melhoria de política (gulosa)
        Q, Pi_new = melhorar_politica(
            env=env,
            V=V,
            gamma=gamma,
        )
        Pi = Pi_new

        # 3) Critério de parada baseado apenas em V
        if np.linalg.norm(V - V_prev, ord=np.inf) < theta:
            break

    return V, Q, Pi, k


# %% [markdown]
# ## Experimento

# %% [markdown]
# ### Simulação

# %%
V, Q, Pi, k = iteracao_de_politica_truncada(
    env,                 # ambiente FrozenLake-v1 (unwrapped)
    gamma=0.95,          # fator de desconto
    j_truncado=100,      # número de varreduras por iteração externa (avaliação truncada)
    theta=1e-8           # critério de parada externo: ||V - V_prev||_inf < theta
)

# V  -> (n_estados,)         valores de estado após convergência do laço externo
# Q  -> (n_estados,n_acoes)  valores de ação da última melhoria de política
# Pi -> (n_estados,n_acoes)  política determinística final
# k  -> int                  número de iterações externas executadas

# %% [markdown]
# ### Visualização

# %%
# Q: ndarray (n_estados, n_acoes)
plot_tabular(Q, kind="Q")

# %%
# Pi: ndarray (n_estados, n_acoes)
plot_tabular(Pi, kind="Pi")

# %%
# V: ndarray (n_estados,)
plot_tabular(V, kind="V", ambiente=env, center_zero=False)

# %%
# Política (setas) sobre o ambiente
visualizar_politica(Pi, env=env)

# %%
env = gym.make("FrozenLake-v1", map_name=map_name, render_mode=render_mode, is_slippery=is_slippery)    # Cria o ambiente FrozenLake
frames = []                                                                                             # Lista que armazenará todos os frames (imagen_estados) do episódio
n_episodios = 5                                                                                         # Número de episódios
for ep in range(n_episodios):
  observation, info = env.reset()                                                                       # Reinicia o ambiente e obtém o primeiro estado (observation)
  for _ in range(100):                                                                                  # Executa um episódio de até 100 passos
      action = int(np.argmax(Pi[observation]))                                                          # Seleciona a ação da política ótima
      observation, reward, terminated, truncated, info = env.step(action)                               # Aplica a ação no ambiente
      frames.append(env.render())                                                                       # Captura a imagem do ambiente após a ação
      if terminated or truncated:                                                                       # Verifica se o episódio acabou (chegou no objetivo ou caiu no buraco)
          break
env.close()                                                                                             # Encerra o ambiente corretamente

# %%
# Salva os frames coletados como um arquivo GIF animado
gif_path="frozenlake.gif"
imageio.mimsave(gif_path, frames, format="GIF", fps=2)

# %%
# Exibe o GIF diretamente no notebook
Image(filename=gif_path)

# %% [markdown]
# # Tarefa
# 
# 1. Implemente o algoritmo **iteração de política truncada**.
# 2. Gere um **gráfico de dispersão** em que cada ponto (x,y) corresponde à (valor do j_truncado, iteração em que a condição de convergência foi satisfeita para este j_truncado).
# 
# ** Utilize a seguinte configuração do ambiente FrozenLake para os experimentos**
# 
# - `map_name = '8x8'` e `map_name = '4x4'`      
# - `render_mode="rgb_array"`
# - `is_slippery=True`
# 
# **No experimento com configuração `map_name = '4x4'` mostrar:**
# 
# 1. **Figuras**:
#    - heatmap de $V(s)$ (função `plot_tabular`);
#    - heatmap de $Q(s,a)$ (função `plot_tabular`);
#    - heatmap de $\pi(a\mid s)$ (função `plot_tabular`);
#    - gráficos de barras de $\pi(a\mid s)$ (função `visualizar_politica`).
#    - gráfico de dispersão
# 
# **No experimento com configuração `map_name = '8x8'` mostrar:**
# 
# 1. **Figura**:
#    - gráfico de dispersão
# 
# **Entregáveis:**
# 
# 2. **Código** (notebook `.ipynb`)
# 1. **Relatório** (`.pdf`).
# - O PDF deve conter:
#   - **Setup** (parâmetros usados).
#   - **Resultados** (figuras e tabelas organizadas por experimento).
#   - **Análises curtas** por experimento.
# - O PDF **NÃO** deve conter:
#     - Códigos.


