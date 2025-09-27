# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import seaborn as sns
from typing import Dict, Tuple, List, Union, Optional, Set


# %%
class AmbienteNavegacaoLabirinto:
    def __init__(
        self,
        world_size: Tuple[int, int],
        bad_states: List[Tuple[int, int]],
        target_states: List[Tuple[int, int]],
        allow_bad_entry: bool = False,
        rewards: Optional[List[float]] = None,
    ) -> None:
        if rewards is None:
            rewards = [-1, -1, 1, 0]

        self.n_rows, self.n_cols = world_size
        self.bad_states = set(bad_states)
        self.target_states = set(target_states)
        self.allow_bad_entry = allow_bad_entry

        for st in self.bad_states | self.target_states:
            if not (0 <= st[0] < self.n_rows and 0 <= st[1] < self.n_cols):
                raise ValueError(f"Estado {st} fora dos limites.")
        if self.bad_states & self.target_states:
            raise ValueError("bad_states e target_states devem ser disjuntos.")

        self.r_boundary = rewards[0]
        self.r_bad = rewards[1]
        self.r_target = rewards[2]
        self.r_other = rewards[3]
        self.action_space = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}
        self.recompensas_possiveis = np.array(sorted(set(rewards)))
        self.reward_map = {r: i for i, r in enumerate(self.recompensas_possiveis)}
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = len(self.action_space)
        self.n_rewards = self.recompensas_possiveis.shape[0]
        self.state_transition_probabilities = np.zeros(
            (self.n_states, self.n_states, self.n_actions)
        )
        self.reward_probabilities = np.zeros(
            (self.n_rewards, self.n_states, self.n_actions)
        )
        self.recompensas_imediatas = np.zeros((self.n_states, self.n_actions))
        self.transicao_de_estados = np.zeros((self.n_states, self.n_actions), dtype=int)
        self.agent_pos = (0, 0)
        self._init_dynamics()

    def __repr__(self) -> str:
        return (
            f"AmbienteNavegacaoLabirinto({self.n_rows}x{self.n_cols}, "
            f"bad={len(self.bad_states)}, target={len(self.target_states)}, "
            f"allow_bad_entry={self.allow_bad_entry}, agent_pos={self.agent_pos})"
        )

    def __str__(self) -> str:
        return self.render(as_string=True)

    def reset(self) -> Tuple[int, int]:
        self.agent_pos = (0, 0)
        return self.agent_pos

    def step(
        self, acao: int, *, linear: bool = False
    ) -> Tuple[Union[int, Tuple[int, int]], float]:
        estado_atual = self.agent_pos
        proposta = self._proposta(estado_atual, acao)
        recompensa = self._compute_reward(proposta)
        destino = self._destino_final(estado_atual, acao)
        self.agent_pos = destino
        proximo_estado = self.state_to_index(destino) if linear else destino
        return proximo_estado, recompensa

    def reset_to_state(
        self,
        estado: Union[Tuple[int, int], int],
        verificar_validade_estado: bool = True,
    ) -> Tuple[int, int]:
        if isinstance(estado, int):
            estado = self.index_to_state(estado)

        if verificar_validade_estado and not self._in_bounds(estado):
            raise ValueError(f"Estado {estado} fora dos limites do labirinto.")

        self.agent_pos = tuple(estado)

        return self.agent_pos

    def is_bad(self, state: Union[int, Tuple[int, int]]) -> bool:
        if isinstance(state, int):
            state = self.index_to_state(state)
        return state in self.bad_states

    def is_target(self, state: Union[int, Tuple[int, int]]) -> bool:
        if isinstance(state, int):
            state = self.index_to_state(state)
        return state in self.target_states

    def state_to_index(self, estado: Tuple[int, int]) -> int:
        linha, coluna = estado
        return linha * self.n_cols + coluna

    def index_to_state(self, indice: int) -> Tuple[int, int]:
        return divmod(indice, self.n_cols)

    def enumerate_states(self) -> List[int]:
        return list(range(self.n_states))

    def enumerate_actions(self) -> List[int]:
        return list(self.action_space.keys())

    def render(
        self,
        *,
        as_string: bool = True,
        show_coords: bool = False,
        legend: bool = True,
        chars: dict | None = None,
    ) -> str:
        if chars is None:
            chars = {"agent": "A", "bad": "B", "target": "T", "empty": "."}

        linhas = []

        if show_coords:
            header = "    " + " ".join(f"{c:2d}" for c in range(self.n_cols))
            linhas.append(header)
            linhas.append("    " + "--" * self.n_cols)

        for r in range(self.n_rows):
            row_syms = []
            for c in range(self.n_cols):
                sym = chars["empty"]
                if (r, c) in self.bad_states:
                    sym = chars["bad"]
                if (r, c) in self.target_states:
                    sym = chars["target"]
                if self.agent_pos == (r, c):
                    sym = chars["agent"]
                row_syms.append(sym)

            linha_str = " ".join(row_syms)
            if show_coords:
                linhas.append(f"{r:2d} | {linha_str}")
            else:
                linhas.append(linha_str)

        if legend:
            linhas.append("")
            linhas.append(
                f"Legenda: {chars['agent']}=agente, {chars['bad']}=bad, "
                f"{chars['target']}=target, {chars['empty']}=vazio"
            )

        out = "\n".join(linhas)
        return out if as_string else print(out)

    def plot_labirinto(
        self, ax=None, titulo: str = "Visualização do Labirinto", cbar: bool = False
    ):
        matriz = np.zeros((self.n_rows, self.n_cols), dtype=int)

        for r, c in self.bad_states:
            matriz[r, c] = 1
        for r, c in self.target_states:
            matriz[r, c] = 2

        cmap = ListedColormap(["white", "red", "green"])

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(self.n_cols, self.n_rows))

        ax = sns.heatmap(
            matriz,
            cmap=cmap,
            cbar=cbar,
            linewidths=0.5,
            linecolor="gray",
            square=True,
            ax=ax,
        )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        for side in ("left", "right", "top", "bottom"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(0.5)
            ax.spines[side].set_edgecolor("gray")

        ax.set_title(titulo)

        if fig is not None:
            plt.tight_layout()
            plt.show()

        return

    def _init_dynamics(self):
        self.recompensas_imediatas.fill(0.0)
        self.transicao_de_estados.fill(0)
        self.state_transition_probabilities.fill(0.0)
        self.reward_probabilities.fill(0.0)

        for s in self.enumerate_states():
            estado_atual = self.index_to_state(s)
            for a in self.enumerate_actions():
                proposta = self._proposta(estado_atual, a)
                r = self._compute_reward(proposta)
                destino = self._destino_final(estado_atual, a)
                s_next = self.state_to_index(destino)

                self.recompensas_imediatas[s, a] = r
                self.transicao_de_estados[s, a] = s_next

                self.state_transition_probabilities[s_next, s, a] = 1.0
                self.reward_probabilities[self.reward_map[r], s, a] = 1.0

    def _proposta(self, state: Tuple[int, int], acao: int) -> Tuple[int, int]:
        dl, dc = self.action_space[acao]
        return (state[0] + dl, state[1] + dc)

    def _destino_final(self, state: Tuple[int, int], acao: int) -> Tuple[int, int]:
        proposta = self._proposta(state, acao)
        if not self._in_bounds(proposta):
            return state
        if (not self.allow_bad_entry) and self.is_bad(proposta):
            return state
        return proposta

    def _in_bounds(self, posicao: Tuple[int, int]) -> bool:
        linha, coluna = posicao
        return 0 <= linha < self.n_rows and 0 <= coluna < self.n_cols

    def _compute_reward(self, destino: Tuple[int, int]) -> float:
        if not self._in_bounds(destino):
            return self.r_boundary
        elif self.is_bad(destino):
            return self.r_bad
        elif self.is_target(destino):
            return self.r_target
        else:
            return self.r_other


# %%
def _prepare_grid(env, ax=None, draw_cells=True):
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
    if isinstance(policy, np.ndarray):
        a_star = np.argmax(policy, axis=1)
        return {env.index_to_state(s): int(a_star[s]) for s in range(env.n_states)}

    sample_val = next(iter(policy.values()))
    if isinstance(sample_val, np.ndarray):
        return {pos: int(np.argmax(probs)) for pos, probs in policy.items()}
    else:
        return policy


def plot_policy(env, policy, ax=None, titulo="Política"):
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
    kind: str = "Q",
    ambiente=None,
    ax=None,
    cbar: bool = True,
    fmt: str = ".1f",
    center_zero: bool = True,
):
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

    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.5)
        ax.spines[side].set_edgecolor("gray")

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
# ## Iteração de valor


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
# ## Experimento

# %% [markdown]
# ### Simulação

# %%
ambiente = AmbienteNavegacaoLabirinto(
    world_size=(5, 5),
    bad_states=[(1, 1), (1, 2), (2, 2), (3, 1), (3, 3), (4, 1)],
    target_states=[(3, 2)],
    allow_bad_entry=True,
    rewards=[-1, -1, 1, 0],
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

# %% [markdown]
# # Tarefa:
#
# 1. Variação do fator de desconto
# - Observar e reportar o efeito de diferentes valores da taxa de desconto (por exemplo: $\gamma \in \{\,0.0,\ 0.5,\ 0.9\,\}$)
# 2. Penalidade de estados ruins mais branda
# - Observar e reportar o efeito de trocar $r_{\text{bad}}=-10$ para $r_{\text{bad}}=-1$.
# 3. Transformação afim nas recompensas
# - Observar e reportar o efeito de uma transformação afim ($r' = a\,r + b$, com $a>0$) em todas as recompensas, isto é, em todos os elementos de $[\,r_{\text{boundary}}, r_{\text{bad}}, r_{\text{target}}, r_{\text{other}}\,]$.
#
# **Configuração base (baseline)**
#
# - `world_size = (5, 5)`
# - `target_states = [(3, 2)]`
# - `bad_states = [(1, 1), (1, 2), (2, 2), (3, 1), (3, 3), (4, 1)]`
# - `allow_bad_entry = True`
# - recompensas base: $[\,r_{\text{boundary}},\ r_{\text{bad}},\ r_{\text{target}},\ r_{\text{other}}\,] = [-1,\ -10,\ 1,\ 0]$
# - tolerância e limite: $\theta = 10^{-6}$, `max_iteracoes = 1000`
#
# > Se alterar qualquer parâmetro do setup, **documente explicitamente** no relatório.
#
# **Em todos os experimentos mostrar:**
#
# 1. **Figuras**:
#    - heatmap de $V(s)$ no grid $(n_{\text{rows}}\times n_{\text{cols}})$;
#    - heatmap de $Q(s,a)$ (ações nas linhas, estados nas colunas);
#    - heatmap de $\pi(a\mid s)$ (probabilidades).
# 2. **Convergência**: número de iterações até $\lVert v_{k+1}-v_k\rVert_\infty < \theta$.
# 3. **Discussão**: texto breve (3–6 linhas) por experimento.
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

# %%
