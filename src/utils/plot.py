
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

def _prepare_grid(env, ax=None, draw_cells=True):
    """
    Configura o grid. Se 'ax' não for passado, cria 'fig, ax'; caso contrário retorna 'fig=None, ax'.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(env.n_cols, env.n_rows))

    ax.set_xlim(0, env.n_cols)
    ax.set_ylim(0, env.n_rows)
    ax.set_xticks(np.arange(0, env.n_cols + 1, 1))
    ax.set_yticks(np.arange(0, env.n_rows + 1, 1))
    ax.grid(True)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    if draw_cells:
        for r in range(env.n_rows):
            for c in range(env.n_cols):
                cell = (r, c)
                if cell in env.bad_states:
                    color = "red"
                elif cell in env.target_states:
                    color = "green"
                else:
                    color = "white"
                rect = patches.Rectangle(
                    (c, r), 1, 1, facecolor=color, edgecolor="gray"
                )
                ax.add_patch(rect)

    return fig, ax


def _coerce_policy(env, policy):
    """
    Normaliza a política para o formato dict[(r,c)] -> ação (int).
    Aceita:
      - dict[(r,c)] -> ação
      - dict[(r,c)] -> vetor de probabilidades
      - Pi ndarray (n_estados, n_acoes)
    """
    # caso 1: matriz Pi (ndarray)
    if isinstance(policy, np.ndarray):
        a_star = np.argmax(policy, axis=1)
        return {env.index_to_state(s): int(a_star[s]) for s in range(env.n_states)}

    # caso 2: dicionário
    sample_val = next(iter(policy.values()))
    if isinstance(sample_val, np.ndarray):
        return {pos: int(np.argmax(probs)) for pos, probs in policy.items()}
    else:
        return policy


def plot_policy(env, policy, ax=None, titulo="Política"):
    """
    Desenha setas/círculos de uma política. 'policy' pode ser:
      - dict[(r,c)] -> ação
      - dict[(r,c)] -> vetor de probabilidades
      - ndarray com shape (n_estados, n_acoes)
    """
    fig, ax = _prepare_grid(env, ax=ax)

    policy_dict = _coerce_policy(env, policy)
    color = "black"
    lw = 1.5

    for (r, c), action in policy_dict.items():
        x, y = c + 0.5, r + 0.5
        if action == 0:  # cima
            ax.arrow(
                x,
                y,
                dx=0,
                dy=-0.3,
                head_width=0.2,
                head_length=0.2,
                fc=color,
                ec=color,
                linewidth=lw,
            )
        elif action == 1:  # baixo
            ax.arrow(
                x,
                y,
                dx=0,
                dy=0.3,
                head_width=0.2,
                head_length=0.2,
                fc=color,
                ec=color,
                linewidth=lw,
            )
        elif action == 2:  # esquerda
            ax.arrow(
                x,
                y,
                dx=-0.3,
                dy=0,
                head_width=0.2,
                head_length=0.2,
                fc=color,
                ec=color,
                linewidth=lw,
            )
        elif action == 3:  # direita
            ax.arrow(
                x,
                y,
                dx=0.3,
                dy=0,
                head_width=0.2,
                head_length=0.2,
                fc=color,
                ec=color,
                linewidth=lw,
            )
        elif action == 4:  # ficar
            circ = patches.Circle(
                (x, y), 0.1, edgecolor=color, facecolor="none", linewidth=lw
            )
            ax.add_patch(circ)

    ax.set_title(titulo)
    if fig is not None:
        plt.tight_layout()
        plt.show()
    return ax


def plot_tabular(
    data,
    kind: str = "Q",  # "Q" (valores de ação), "Pi" (política), "V" (valores de estado)
    ambiente=None,  # necessário quando kind="V" para reshape
    ax=None,
    cbar: bool = True,
    fmt: str = ".1f",
    center_zero: bool = True,  # só relevante para "Q" e "V"
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
    ylabel = {"V": "Linhas", "PI": "Ações", "Q": "Ações"}
    title = {
        "V": "Valores de Estado (V(s))",
        "PI": r"Política ($\pi(a|s)$ transposta)",
        "Q": "Valores de ação (Q(s, a) transposta)",
    }

    fig = None

    #  V(s): precisa do shape do grid
    match kind:
        case "V":

            if ambiente is None:
                raise ValueError(
                    "Para kind='V', passe 'ambiente' para reshape (n_rows, n_cols)."
                )

            M = data.reshape(ambiente.n_rows, ambiente.n_cols)

            if ax is None:
                fig, ax = plt.subplots(figsize=(ambiente.n_cols, ambiente.n_rows))

            if center_zero:
                vmax = float(np.abs(M).max())
                vmin = -vmax
            else:
                vmin = float(M.min())
                vmax = float(M.max())

            cmap, square = "bwr", True

        case "PI" | "Q":

            # Q(s,a) e Pi(a|s): ações nas linhas, estados nas colunas
            M = (
                data.T
            )  # data: (n_estados, n_acoes) -> transposto para (n_acoes, n_estados)
            n_acoes, n_estados = M.shape

            if ax is None:
                fig, ax = plt.subplots(figsize=(n_estados, n_acoes))

            if kind == "PI":
                cmap = "Blues"
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
        ax=ax,
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

def plot_visitas_log(n_visitas):
    """
    Gera um gráfico de dispersão com escala logarítmica no eixo y
    mostrando o número de visitas para cada par (s,a).

    Parâmetros
    ----------
    n_visitas : np.ndarray
        Matriz de número de visitas de shape (n_states, n_actions).
    """
    n_states, n_actions = n_visitas.shape
    x = np.arange(n_states * n_actions)  # índice linear do par (s,a)
    y = n_visitas.flatten()              # número de visitas

    plt.figure(figsize=(8, 4))
    plt.scatter(x, y, s=10, alpha=0.7)
    plt.yscale('log')
    plt.xlabel("Índice linear do par (s,a)")
    plt.ylabel("Número de visitas ao par (s,a)")
    plt.title("Frequência de visitas (escala log)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()