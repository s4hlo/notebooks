# %% [markdown]
# # Laboratório 10: Sarsa com aproximação de função (FrozenLake)

# %% [markdown]
# ## Importações

# %%
# # Instala os pacotes necessários:
# # - gymnasium[toy-text]: inclui ambientes simples como FrozenLake, Taxi, etc.
# # - imageio[ffmpeg]: permite salvar vídeos e GIFs (formato .mp4 ou .gif)
!pip install gymnasium[toy-text] imageio[ffmpeg]

# %%
import gymnasium as gym
import imageio
from IPython.display import Image
import numpy as np
from typing import Dict, Tuple, List, Union, Optional, Set
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import seaborn as sns
import pandas as pd
from tqdm.auto import tqdm
from sklearn.neural_network import MLPRegressor
from dataclasses import dataclass

# %% [markdown]
# ## Funções auxiliares para visualização

# %%
def plotar_metricas(
    episodio_len: list[int],
    episodio_return: list[float],
    td_errors: list[float],
    eps_hist: list[float],
    janela: int = 100
) -> None:
    """
    Plota métricas de treinamento:
      - tamanho do episódio (com média móvel)
      - retorno por episódio (com média móvel)
      - TD-error médio (com média móvel)
      - epsilon (SEM média móvel)
    """

    df = pd.DataFrame({
        'episodio': np.arange(len(episodio_len)),
        'tamanho': episodio_len,
        'retorno': episodio_return,
        'td_error': td_errors,
        'epsilon': eps_hist
    })

    df['tamanho_ma'] = df['tamanho'].rolling(window=janela).mean()
    df['retorno_ma'] = df['retorno'].rolling(window=janela).mean()
    df['td_ma']      = df['td_error'].rolling(window=janela).mean()

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 10), sharex=True)

    # --- Episódio length ---
    sns.lineplot(data=df, x='episodio', y='tamanho', ax=axs[0,0], alpha=0.3, label='Tamanho')
    sns.lineplot(data=df, x='episodio', y='tamanho_ma', ax=axs[0,0], label=f'Média móvel ({janela})')
    axs[0,0].set_ylabel('Passos')
    axs[0,0].set_title("Tamanho do episódio")
    axs[0,0].grid()
    axs[0,0].legend()

    # --- Retorno ---
    sns.lineplot(data=df, x='episodio', y='retorno', ax=axs[0,1], alpha=0.3, label='Retorno', color='orange')
    sns.lineplot(data=df, x='episodio', y='retorno_ma', ax=axs[0,1], label=f'Média móvel ({janela})', color='red')
    axs[0,1].set_title("Retorno por episódio")
    axs[0,1].set_ylabel('Retorno total')
    axs[0,1].grid()
    axs[0,1].legend()

    # --- TD-error ---
    sns.lineplot(data=df, x='episodio', y='td_error', ax=axs[1,0], alpha=0.3, label='TD-error', color='purple')
    sns.lineplot(data=df, x='episodio', y='td_ma', ax=axs[1,0], label=f'Média móvel ({janela})', color='black')
    axs[1,0].set_title("TD-error")
    axs[1,0].set_ylabel('TD-error')
    axs[1,0].grid()
    axs[1,0].legend()

    # --- Epsilon ---
    sns.lineplot(data=df, x='episodio', y='epsilon', ax=axs[1,1], label='ε', color='green')
    axs[1,1].set_title("Epsilon (ε)")
    axs[1,1].set_ylabel('ε')
    axs[1,1].set_xlabel('Episódio')
    axs[1,1].grid()
    axs[1,1].legend()

    plt.tight_layout()
    plt.show()

# %%
def visualizar_politica(
    Pi: np.ndarray,
    env_name: str,
    *,
    action_labels: Optional[List[str]] = None,
    destacar_gulosa: bool = True,
    suptitle: Optional[str] = "Política (distribuições por estado)",
    **env_kwargs,
) -> None:
    """
    Visualiza a política como barras por estado.

    Parâmetros
    ----------
    Pi : np.ndarray, shape (n_estados, n_acoes)
        Política estocástica (cada linha soma ≈ 1).
    env_name : {"FrozenLake-v1","CliffWalking-v1"}
        Nome do ambiente Gymnasium.
    action_labels : list[str] | None
        Rótulos das ações. Se None, usa o mapeamento correto de cada ambiente.
    destacar_gulosa : bool
        Se True, escurece a barra da ação de maior probabilidade em cada estado.
    suptitle : str | None
        Título superior da figura.
    **env_kwargs :
        Parâmetros repassados ao gym.make (ex.: map_name="4x4", is_slippery=False).
    """

    n_estados, n_acoes = Pi.shape

    env = gym.make(env_name, **env_kwargs)

    try:
        # --------- Grid e terminais ---------
        if env_name == "FrozenLake-v1":
            desc = env.unwrapped.desc
            desc_str = np.char.decode(desc, "utf-8") if getattr(desc.dtype, "kind", "") == "S" else desc.astype(str)
            n_rows, n_cols = desc_str.shape
            holes = (desc_str == "H")
            goal  = (desc_str == "G")
            default_labels = ["←", "↓", "→", "↑"]   # 0:left,1:down,2:right,3:up

        elif env_name == "CliffWalking-v1":
            n_rows, n_cols = env.unwrapped.shape
            holes = np.zeros((n_rows, n_cols), dtype=bool)
            holes[n_rows - 1, 1:n_cols - 1] = True  # falésia
            goal = np.zeros((n_rows, n_cols), dtype=bool)
            goal[n_rows - 1, n_cols - 1] = True     # alvo
            default_labels = ["↑", "→", "↓", "←"]   # 0:up,1:right,2:down,3:left

        else:
            raise ValueError("env_name deve ser 'FrozenLake-v1' ou 'CliffWalking-v1'.")

        if n_rows * n_cols != n_estados:
            raise ValueError(f"Incompatibilidade: grid {n_rows}x{n_cols} != n_estados={n_estados} em Pi.")

        # --------- Rótulos das ações ---------
        if action_labels is None:
            if n_acoes == 4:
                action_labels = default_labels
            else:
                action_labels = [f"a{i}" for i in range(n_acoes)]
        elif len(action_labels) != n_acoes:
            action_labels = [f"a{i}" for i in range(n_acoes)]

        # --------- Figura ---------
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.2 * n_rows))
        axs = np.array(axs).reshape(-1)

        for s in range(n_estados):
            r, c = divmod(s, n_cols)
            ax = axs[s]

            # Estados terminais (sem barras)
            if holes[r, c]:
                # vermelho translúcido
                ax.set_facecolor((1.0, 0.0, 0.0, 0.18))
                ax.set_title(f"Estado {s} ({'H' if env_name=='FrozenLake-v1' else 'penhasco'})")
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values(): sp.set_visible(True)
                continue

            if goal[r, c]:
                # verde translúcido
                ax.set_facecolor((0.0, 1.0, 0.0, 0.18))
                ax.set_title(f"Estado {s} ({'G' if env_name=='FrozenLake-v1' else 'alvo'})")
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values(): sp.set_visible(True)
                continue

            # Estado não-terminal: barras da distribuição
            pi = Pi[s].astype(float)
            if pi.sum() > 0:
                pi /= pi.sum()  # normalização defensiva

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

        for i in range(n_estados, len(axs)):
            fig.delaxes(axs[i])

        if suptitle:
            fig.suptitle(suptitle, y=1.02, fontsize=12)

        plt.tight_layout()
        plt.show()
    finally:
        env.close()

# %%
def plot_V_grid(
    V: np.ndarray,
    env_name: str,
    *,
    ax=None,
    cbar: bool = True,
    fmt: str = ".1f",
    center_zero: bool = False,
    title: str | None = None,
):
    """
    Plota V(s) em formato de grid, com reshape automático para ambientes discretos padrão.

    Regras:
      - Se V for 2D: plota diretamente (caso 'Labirinto' ou outro ambiente customizado).
      - Se env_name == 'CliffWalking-v1': reshape para (4, 12).
      - Se env_name == 'FrozenLake-v1': infere lado = sqrt(n_estados) → (lado, lado).

    Parâmetros
    ----------
    V : np.ndarray
        Vetor 1D (n_estados,) ou matriz 2D (n_rows, n_cols) de valores de estado.
    env_name : str
        Nome do ambiente ('FrozenLake-v1', 'CliffWalking-v1' ou outro).
    """
    # Inferir formato
    if V.ndim == 2:
        M = V
    else:
        if env_name == "CliffWalking-v1":
            M = V.reshape(4, 12)
        elif env_name == "FrozenLake-v1":
            n = V.shape[0]
            side = int(np.sqrt(n))
            if side * side != n:
                raise ValueError(f"FrozenLake: n_estados={n} não forma um grid quadrado.")
            M = V.reshape(side, side)
        else:
            raise ValueError(
                f"Para env_name='{env_name}', V deve ser passado já como array 2D."
            )

    # Preparar figura
    if ax is None:
        fig, ax = plt.subplots(figsize=(M.shape[1], M.shape[0]))
    else:
        fig = None

    if center_zero:
        vmax = float(np.abs(M).max())
        vmin = -vmax
    else:
        vmin = float(M.min())
        vmax = float(M.max())

    # Plotar heatmap
    sns.heatmap(
        M, annot=True, fmt=fmt, cmap="bwr",
        vmin=vmin, vmax=vmax, cbar=cbar, square=True,
        linewidths=0.5, linecolor="gray", ax=ax
    )

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("Colunas"); ax.set_ylabel("Linhas")
    ax.set_title(title if title else f"V(s) — {env_name}")

    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.5)
        ax.spines[side].set_edgecolor("gray")

    if fig is not None:
        plt.tight_layout()
        plt.show()

    return ax

def plot_tabular(
    data: np.ndarray,
    kind: str = "Q",                 # "Q", "Pi" ou "V"
    env_name: str | None = None,     # necessário quando kind="V"
    ax=None,
    cbar: bool = True,
    fmt: str = ".1f",
    center_zero: bool = True,        # só relevante para "Q" e "V"
    title: str | None = None,
):
    """
    Plota matrizes tabulares de RL:
      - kind="Q": heatmap de Q(s,a) (ações nas linhas, estados nas colunas).
      - kind="Pi": heatmap de π(a|s) (probabilidades) (ações nas linhas, estados nas colunas).
      - kind="V": usa plot_V_grid para reorganizar e plotar V(s) no grid conforme env_name.

    Para kind="V":
      - env_name == "CliffWalking-v1": reshape automático para (4,12).
      - env_name == "FrozenLake-v1": inferência automática do lado (√n_estados).
      - qualquer outro env_name: espera V já 2D.
    """
    kind = kind.upper()

    if kind == "V":
        if env_name is None:
            raise ValueError("Para kind='V', passe env_name ('FrozenLake-v1', 'CliffWalking-v1' ou outro).")
        # delega ao plot_V_grid
        return plot_V_grid(
            V=data,
            env_name=env_name,
            ax=ax,
            cbar=cbar,
            fmt=fmt,
            center_zero=center_zero,
            title=title if title else f"V(s) — {env_name}",
        )

    # casos Q e Pi
    xlabel = {"PI": "Estados", "Q": "Estados"}
    ylabel = {"PI": "Ações",  "Q": "Ações"}
    default_title = {"PI": r"Política ($\pi(a|s)$) transposta", "Q": "Valores de ação (Q) transposta"}

    # Q(s,a) e Pi(a|s): ações nas linhas, estados nas colunas
    M = data.T  # (n_estados, n_acoes) -> (n_acoes, n_estados)
    n_acoes, n_estados = M.shape

    if ax is None:
        fig, ax = plt.subplots(figsize=(n_estados, n_acoes))
    else:
        fig = None

    if kind == "PI":
        cmap = "Blues"; vmin, vmax = 0.0, 1.0
    elif kind == "Q":
        cmap = "bwr"
        if center_zero:
            vmax = float(np.abs(M).max()); vmin = -vmax
        else:
            vmin = float(M.min()); vmax = float(M.max())
    else:
        raise ValueError(f"kind desconhecido: {kind!r} (use 'Q', 'Pi' ou 'V').")

    ax = sns.heatmap(
        data=M,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar=cbar,
        square=False,
        linewidths=0.5,
        linecolor="gray",
        ax=ax
    )

    ax.set_xlabel(xlabel[kind])
    ax.set_ylabel(ylabel[kind])
    ax.set_title(title if title else default_title[kind])

    # bordas e rótulos
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.5)
        ax.spines[side].set_edgecolor("gray")

    ax.set_xticks(np.arange(n_estados) + 0.5)
    ax.set_xticklabels([f"s{i}" for i in range(n_estados)], rotation=0)
    ax.set_yticks(np.arange(n_acoes) + 0.5)
    ax.set_yticklabels([f"a{i}" for i in range(n_acoes)], rotation=0)

    if fig is not None:
        plt.tight_layout()
        plt.show()

    return ax

# %%
# helpers de grid/máscaras
def _grid_info_from_env(env_name: str, map_name: str | None = None, is_slippery: bool = False):
    """
    Retorna (n_rows, n_cols, holes_set, cliffs_set, goals_set, start_rc)
    para FrozenLake-v1 ou CliffWalking-v1.
    """
    holes, cliffs, goals = set(), set(), set()
    start_rc = None

    if env_name == "FrozenLake-v1":
        kwargs = {}
        if map_name is not None:
            kwargs["map_name"] = map_name
        kwargs["is_slippery"] = is_slippery

        env = gym.make("FrozenLake-v1", **kwargs)
        desc = env.unwrapped.desc
        # decode se dtype for bytes
        desc = np.array(desc, dtype=str) if desc.dtype.kind != "S" else np.char.decode(desc, "utf-8")
        n_rows, n_cols = desc.shape
        for r in range(n_rows):
            for c in range(n_cols):
                ch = desc[r, c]
                if ch == "H": holes.add((r, c))
                elif ch == "G": goals.add((r, c))
                elif ch == "S": start_rc = (r, c)
        if start_rc is None:
            start_rc = (0, 0)
        env.close()
        return n_rows, n_cols, holes, cliffs, goals, start_rc

    elif env_name == "CliffWalking-v1":
        n_rows, n_cols = 4, 12
        start_rc = (n_rows - 1, 0)
        goal_rc  = (n_rows - 1, n_cols - 1)
        goals.add(goal_rc)
        # penhasco: última linha, colunas 1..10
        for c in range(1, n_cols - 1):
            cliffs.add((n_rows - 1, c))
        return n_rows, n_cols, holes, cliffs, goals, start_rc

    else:
        raise ValueError("env_name deve ser 'FrozenLake-v1' ou 'CliffWalking-v1'.")


def _rc_from_state(s: int, n_cols: int) -> tuple[int, int]:
    return divmod(s, n_cols)  # (r,c)


# simulação de trajetória
def simular_trajetoria_gym(
    Pi: np.ndarray,
    env_name: str,
    *,
    map_name: str | None = None,
    is_slippery: bool = False,
    max_steps: int = 200
):
    """
    Executa uma trajetória determinística (gulosa em Pi) no Gym (FrozenLake/CliffWalking).
    Retorna:
      - estados: lista de (r,c)
      - acoes: lista de ints
      - recompensas: lista de floats
    """
    # cria env
    if env_name == "FrozenLake-v1":
        kwargs = {}
        if map_name is not None:
            kwargs["map_name"] = map_name
        kwargs["is_slippery"] = is_slippery
        env = gym.make("FrozenLake-v1", **kwargs)
    elif env_name == "CliffWalking-v1":
        env = gym.make("CliffWalking-v1")
    else:
        raise ValueError("env_name deve ser 'FrozenLake-v1' ou 'CliffWalking-v1'.")

    n_rows, n_cols, holes, cliffs, goals, start_rc = _grid_info_from_env(env_name, map_name, is_slippery)

    # sanity check de dimensões
    n_states, n_actions = Pi.shape
    assert n_states == n_rows * n_cols, f"Pi.shape[0]={n_states} != n_rows*n_cols={n_rows*n_cols}"
    assert n_actions == env.action_space.n, f"Pi.shape[1]={n_actions} != action_space.n={env.action_space.n}"

    state, _ = env.reset()
    # se o env permitir setar estado diretamente, tenta (FrozenLake/cliff costumam expor .unwrapped.s)
    if hasattr(env.unwrapped, "s"):
        env.unwrapped.s = int(state)

    estados_rc = [_rc_from_state(int(state), n_cols)]
    acoes, recompensas = [], []

    for _ in range(max_steps):
        s = int(state)
        a = int(np.argmax(Pi[s]))
        next_state, reward, terminated, truncated, _ = env.step(a)

        acoes.append(a)
        recompensas.append(float(reward))
        estados_rc.append(_rc_from_state(int(next_state), n_cols))

        state = next_state
        if terminated or truncated:
            break

    env.close()
    return estados_rc, acoes, recompensas


# --------------------------
# plot da trajetória no grid
# --------------------------
def plot_trajetoria_gym(
    env_name: str,
    estados: list[tuple[int, int]],
    *,
    map_name: str | None = None,
    is_slippery: bool = False,
    titulo: str = "Trajetória gulosa (Gym)",
    gradiente_temporal: bool = True,
    mostrar_setas: bool = True
):
    """
    Plota a trajetória sobre um grid para FrozenLake/CliffWalking com destaques de buracos/penhasco/goal.
    """
    n_rows, n_cols, holes, cliffs, goals, start_rc = _grid_info_from_env(env_name, map_name, is_slippery)

    # figura/base do grid
    fig, ax = plt.subplots(figsize=(n_cols, n_rows))
    ax.set_xlim(0, n_cols); ax.set_ylim(0, n_rows)
    ax.set_xticks(np.arange(0, n_cols + 1, 1))
    ax.set_yticks(np.arange(0, n_rows + 1, 1))
    ax.grid(True); ax.set_aspect('equal'); ax.invert_yaxis()

    # células coloridas
    for r in range(n_rows):
        for c in range(n_cols):
            cell = (r, c)
            if env_name == "FrozenLake-v1":
                if cell in holes:
                    color = (1.0, 0.0, 0.0, 0.25)  # vermelho translúcido
                elif cell in goals:
                    color = (0.0, 1.0, 0.0, 0.25)  # verde translúcido
                elif cell == start_rc:
                    color = (1.0, 1.0, 0.0, 0.18)  # amarelo leve
                else:
                    color = 'white'
            else:  # CliffWalking
                if cell in cliffs:
                    color = (1.0, 0.0, 0.0, 0.25)
                elif cell in goals:
                    color = (0.0, 1.0, 0.0, 0.25)
                elif cell == start_rc:
                    color = (1.0, 1.0, 0.0, 0.18)
                else:
                    color = 'white'

            rect = patches.Rectangle((c, r), 1, 1, facecolor=color, edgecolor='gray')
            ax.add_patch(rect)

    # extrai xs, ys (centros)
    xs = [c + 0.5 for (_, c) in estados]
    ys = [r + 0.5 for (r, _) in estados]

    # linha com gradiente temporal
    if gradiente_temporal and len(xs) > 1:
        pontos = np.array([xs, ys]).T
        segmentos = np.stack([pontos[:-1], pontos[1:]], axis=1)
        lc = LineCollection(segmentos, linewidths=2.5)
        cores = np.linspace(0, 1, len(segmentos))
        lc.set_array(cores)
        ax.add_collection(lc)
        cbar = plt.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Progresso temporal")
    else:
        ax.plot(xs, ys, '-o', linewidth=2.5, markersize=4)

    # início/fim
    ax.scatter(xs[0], ys[0], marker='*', s=220, edgecolor='black', facecolor='yellow', zorder=5, linewidths=1.2, label='Início')
    ax.scatter(xs[-1], ys[-1], marker='o', s=150, edgecolor='black', facecolor='none', zorder=5, linewidths=1.2, label='Fim')

    # setas entre células
    if mostrar_setas:
        for i in range(len(xs) - 1):
            dx, dy = xs[i+1] - xs[i], ys[i+1] - ys[i]
            ax.arrow(xs[i], ys[i], dx*0.85, dy*0.85, head_width=0.15, head_length=0.15,
                     length_includes_head=True, fc='black', ec='black', alpha=0.8)

    ax.set_title(titulo)
    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.show()

# %%
def gerar_gif_simulacao(
    Pi: np.ndarray,
    env,
    path_gif: str,
    n_episodios: int = 5,
    max_steps: int = 200,
    fps: int = 2,
    greedy: bool = False,
    normalizar: bool = True,
) -> str:
    """
    Simula episódios com a política Pi (array numpy) e gera um GIF.

    Parâmetros
    ----------
    Pi : np.ndarray, shape (n_estados, n_acoes)
        Política estocástica por estado. Se `greedy=False`, amostra ações segundo Pi[s].
        Se `greedy=True`, escolhe argmax_a Pi[s, a].
    env : gymnasium.Env
        Ambiente Gymnasium com `render_mode='rgb_array'`.
   path_gif : str
        Caminho/arquivo de saída do GIF (ex.: "execucao.gif").
    n_episodios : int, default=5
        Quantidade de episódios simulados no GIF.
    max_steps : int, default=200
        Máximo de passos por episódio.
    fps : int, default=2
        Frames por segundo do GIF.
    greedy : bool, default=False
        Se True, usa ação gulosa (argmax). Se False, amostra segundo Pi[s].
    normalizar : bool, default=True
        Se True, normaliza Pi[s] para somar 1 quando for amostrar.

    Retorna
    -------
    str
        Caminho do GIF gerado.
    """
    # Verificações rápidas para espaços discretos
    assert hasattr(env.observation_space, "n") and hasattr(env.action_space, "n"), \
        "Este utilitário espera espaços Discrete (ex.: FrozenLake, CliffWalking)."
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    if Pi.shape != (n_states, n_actions):
        raise ValueError(f"Shape incompatível: Pi{Pi.shape} != ({n_states}, {n_actions})")

    frames = []

    for _ in range(n_episodios):
        state, _ = env.reset()

        # captura frame inicial do episódio
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        for _ in range(max_steps):
            if greedy:
                action = int(np.argmax(Pi[state]))
            else:
                probs = Pi[state].astype(float)
                if normalizar:
                    s = probs.sum()
                    if s > 0:
                        probs = probs / s
                # fallback se a linha for toda zero (ou numérica inválida)
                if not np.isfinite(probs).all() or probs.sum() <= 0:
                    probs = np.full(n_actions, 1.0 / n_actions)
                action = int(np.random.choice(n_actions, p=probs))

            state, reward, terminated, truncated, _ = env.step(action)

            frame = env.render()
            if frame is not None:
                frames.append(frame)

            if terminated or truncated:
                break

    env.close()
    imageio.mimsave(path_gif, frames, format="GIF", fps=fps)
    return path_gif

# %% [markdown]
# ## Ambiente: nova instância

# %%
# Configura o ambiente
ambiente = 'FrozenLake-v1'
render_mode = 'rgb_array'  # retorna imagens do ambiente como arrays de pixels
map_name = '4x4'
is_slippery = False
env = gym.make(ambiente, map_name=map_name, is_slippery=is_slippery, render_mode=render_mode)
n_states  = env.observation_space.n
n_actions = env.action_space.n
actions_desc = ["left", "down", "right", "up"]

# %% [markdown]
# ## Algoritmo: Sarsa com aproximação de função

# %% [markdown]
# ### Vetor de características: one-hot-encoding (codificação 1-de-c)

# %%
I = np.eye(n_states, dtype=np.float32)
def one_hot(s):
    return I[s:s+1]  # fatia (slice) de I não uma cópia (ex.: I[[s]]) -> mais rápido

# %% [markdown]
# ### Atualização da rede neural

# %%
def _atualiza_Q(net, experiencia, gamma, done):

    s, a, r, s_next, a_next = experiencia

    q_s      = net.predict(one_hot(s))[0]
    q_s_next = net.predict(one_hot(s_next))[0]

    td_target = r if done else (r + gamma * q_s_next[a_next])
    td_error  = td_target - q_s[a]
    target    = q_s.copy()
    target[a] = td_target

    net.partial_fit(X=one_hot(s), y=target.reshape(1, -1))

    return float(td_error)

# %% [markdown]
# ### Sarsa com approximação de função   

# %%
def sarsa_nn(
    env: gym.Env,
    gamma: float = 0.95,
    N: int = 5000,
    start_epsilon: float = 1.0,
    final_epsilon: float = 0.10,
    epsilon_mul: float = 0.9999,
    *,
    hidden: tuple = (32,),
    lr_opt: float = 0.05,
    activation: str = "relu",
    solver: str = "sgd",
    momentum: float = 0.0,
    shuffle: bool = False,
    mlp_random_state: Optional[int] = 0
):
    """
    SARSA com aproximação de função (MLPRegressor).

    Returns:
      Q_hat, Pi_final, visitas, N, episodio_T, episodio_G, td_errors, epsilon_history, net
    """

    # Atalhos
    n_states  = env.observation_space.n
    n_actions = env.action_space.n

    # Rede neural
    net = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation=activation,
        solver=solver,
        learning_rate_init=lr_opt,
        momentum=momentum,
        max_iter=1,
        warm_start=True,
        shuffle=shuffle,
        random_state=mlp_random_state
    )
    net.fit(one_hot(0), np.zeros((1, n_actions), dtype=np.float32))

    # Tabelas
    numero_de_visitas = np.zeros((n_states, n_actions), dtype=float)

    episodio_T, episodio_G, episodio_TD_errors, epsilon_history = [], [], [], []

    epsilon = float(start_epsilon)

    # ε-greedy without storing a global Pi
    def choose_action(q_s: np.ndarray, eps: float) -> int:
        if np.random.rand() < eps:
            return np.random.randint(n_actions)
        return int(np.argmax(q_s))

    pbar = tqdm(range(1, N + 1), desc="Episódios (SARSA com approximação de função)", leave=True)

    for _ in pbar:
        # Reset (s0)
        s, _ = env.reset()

        # Ação inicial a0 ~ Pi(s0)
        q_s = net.predict(one_hot(s))[0]
        a = choose_action(q_s, epsilon)

        G, t, done, td_errors  = 0.0, 0, False, []

        while not done:
            numero_de_visitas[s, a] += 1.0

            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            G += r
            t += 1

            # Próxima ação a_{t+1} ~ Pi(s_{t+1}) (seguindo a política: on-policy)
            q_s_next = net.predict(one_hot(s_next))[0]
            a_next = choose_action(q_s_next, epsilon) if not done else 0

            # Dados
            experiencia = (s, a, r, s_next, a_next)

            # Etapa da estimação de Q(s,a): utiliza aprendizado por diferenças temporais (atualização SARSA)
            td = _atualiza_Q(net, experiencia, gamma, done)
            td_errors.append(td)

            s, a = s_next, a_next

        episodio_T.append(t)
        episodio_G.append(G)
        episodio_TD_errors.append(np.array(td_errors).mean())
        epsilon_history.append(epsilon)

        # tqdm display update (reward, epsilon)
        pbar.set_postfix({"R": f"{G:.3f}", "ε": f"{epsilon:.3f}"})

        # Decaimento de epsilon
        epsilon = max(final_epsilon, epsilon * epsilon_mul)

    # Predição Q(s,·) final
    Q = net.predict(I)

    # Política ε-gulosa final
    Pi = np.full((n_states, n_actions), epsilon / n_actions, dtype=float)
    acoes_gulosas = np.argmax(Q, axis=1)
    Pi[np.arange(n_states), acoes_gulosas] += 1.0 - epsilon

    return Q, Pi, numero_de_visitas, N, episodio_T, episodio_G, episodio_TD_errors, epsilon_history, net

# %% [markdown]
# ## Experimento

# %% [markdown]
# ### Simulação

# %%
# Hiperparâmetros
sarsa_config = dict(
    env = env,
    gamma = 0.9,
    N = 5000,               # número de episódios
    start_epsilon = 1.0,
    final_epsilon = 0.1,
    epsilon_mul = 0.9999,    # decaimento multiplicativo
    hidden = (16,) if map_name == "4x4" else (32,),          # arquitetura MLP
    lr_opt = 0.05,           # taxa de aprendizagem do MLP
    activation = "relu",
    solver = "sgd",
    momentum = 0.0,
    shuffle = False,
    mlp_random_state = 0
)

# %%
Q, Pi, visitas, N, T_hist, G_hist, td_errors, eps_hist, net = sarsa_nn(**sarsa_config)

# %%
# Derivar V a partir de Q:
V = np.sum(Pi * Q, axis=1)

# %% [markdown]
# ### Visualização

# %%
# plotar_metricas(T_hist, G_hist)
plotar_metricas(T_hist, G_hist, td_errors, eps_hist, janela=200)


# %%
visualizar_politica(Pi, ambiente, map_name=map_name, is_slippery=is_slippery)

# %%
estados, acoes, recompensas = simular_trajetoria_gym(Pi, ambiente, map_name=map_name, is_slippery=is_slippery, max_steps=200)
plot_trajetoria_gym(ambiente, estados, map_name=map_name, is_slippery=is_slippery, titulo="Trajetória (gulosa)")


# %%
# Renderiza o ambiente
env.reset()
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
_ = plot_tabular(Pi, kind="Pi")

# %%
# V: ndarray (n_estados,)
print(ambiente)
_ = plot_tabular(V, kind="V", env_name=ambiente, center_zero=False)

# %% [markdown]
# # Tarefa
# 
# Formule e execute 2 experimentos distintos em que apenas um hiperparâmetro é variado enquanto todos os outros são mantidos constatntes. Em seguida compare o desempenho (recompensa média e comprimento de episódio) entre as configurações testadas. Mostre também o erro TD e o decaimento de $\epsilon$.
# 
# Cada um dos 2 experimentos deve:
# - Manter todos os outros parâmetros fixos
# - Coletar métricas de recompensa média e comprimento/estabilidade do episódio
# - Analisar gráficos comparativos e discorrer sobre como a alteração impactou o aprendizado.
# 
# Hiperparâmetros a serem variados (escolha 2 dentre os hiperparâmetros listados abaixo):
# - lr_opt : taxa de aprendizagem da rede neural
# - N: Número de episódios
# - hidden: arquitetura da rede neural
# - epsilon_mul: decaimento de epsilon
# 
# Realize os experimentos com is_slippery=True e is_slippery=False.
# 
# **Entregáveis:**
# 
# 2. **Código** (notebook `*.ipynb`)
# 1. **Relatório** (`*.pdf`).
# - O PDF deve conter:
#   - **Setup** (parâmetros usados).
#   - **Resultados** (figuras e tabelas organizadas por experimento).
#   - **Análises curtas** por experimento.
# - O PDF **NÃO** deve conter:
#     - Códigos.


