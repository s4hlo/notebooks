
# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from typing import Tuple, List, Union, Optional


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
