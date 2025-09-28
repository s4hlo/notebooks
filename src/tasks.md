# iteracao de valor

1.  Variação do fator de desconto

- Observar e reportar o efeito de diferentes valores da taxa de desconto (por exemplo: $\gamma \in \{\,0.0,\ 0.5,\ 0.9\,\}$)

2.  Penalidade de estados ruins mais branda

- Observar e reportar o efeito de trocar $r_{\text{bad}}=-10$ para $r_{\text{bad}}=-1$.

3.  Transformação afim nas recompensas

- Observar e reportar o efeito de uma transformação afim ($r' = a\,r + b$, com $a>0$) em todas as recompensas, isto é, em todos os elementos de $[\,r_{\text{boundary}}, r_{\text{bad}}, r_{\text{target}}, r_{\text{other}}\,]$.

**Configuração base (baseline)**

- `world_size = (5, 5)`
- `target_states = [(3, 2)]`
- `bad_states = [(1, 1), (1, 2), (2, 2), (3, 1), (3, 3), (4, 1)]`
- `allow_bad_entry = True`
- recompensas base: $[\,r_{\text{boundary}},\ r_{\text{bad}},\ r_{\text{target}},\ r_{\text{other}}\,] = [-1,\ -10,\ 1,\ 0]$
- tolerância e limite: $\theta = 10^{-6}$, `max_iteracoes = 1000`

> Se alterar qualquer parâmetro do setup, **documente explicitamente** no relatório.

**Em todos os experimentos mostrar:**

1.  **Figuras**:
    - heatmap de $V(s)$ no grid $(n_{\text{rows}}\times n_{\text{cols}})$;
    - heatmap de $Q(s,a)$ (ações nas linhas, estados nas colunas);
    - heatmap de $\pi(a\mid s)$ (probabilidades).
2.  **Convergência**: número de iterações até $\lVert v_{k+1}-v_k\rVert_\infty < \theta$.
3.  **Discussão**: texto breve (3–6 linhas) por experimento.

**Entregáveis:**

2.  **Código** (notebook `.ipynb`)
1.  **Relatório** (`.pdf`).

- O PDF deve conter:
  - **Setup** (parâmetros usados).
  - **Resultados** (figuras e tabelas organizadas por experimento).
  - **Análises curtas** por experimento.
- O PDF **NÃO** deve conter:
  - Códigos.

# iteracao de politica


1.  Implemente o algoritmo **iteração de política**.
1.  Compare os algoritmos de iteração de valor (laboratório 1) e de iteração de política (laboratório 2) quanto ao número de iterações utilizadas até a condição de convergência ser satisfeita.

**Configuração base (baseline)**

- `world_size = (5, 5)`
- `target_states = [(3, 2)]`
- `bad_states = [(1, 1), (1, 2), (2, 2), (3, 1), (3, 3), (4, 1)]`
- `allow_bad_entry = True`
- recompensas base: $[\,r_{\text{boundary}},\ r_{\text{bad}},\ r_{\text{target}},\ r_{\text{other}}\,] = [-1,\ -10,\ 1,\ 0]$
- tolerância e limite: $\theta = 10^{-6}$, `max_iteracoes = 1000`

> Se alterar qualquer parâmetro do setup, **documente explicitamente** no relatório.

**Em todos os experimentos mostrar:**

1.  **Figuras**:
    - heatmap de $V(s)$ no grid $(n_{\text{rows}}\times n_{\text{cols}})$;
    - heatmap de $Q(s,a)$ (ações nas linhas, estados nas colunas);
    - heatmap de $\pi(a\mid s)$ (probabilidades).
2.  **Convergência**: número de iterações até $\lVert v_{\pi_k}-v_{\pi_{k-1}}\rVert_\infty < \theta_{política}$.
3.  **Discussão**: texto breve (3-6 linhas) por experimento.

**Entregáveis:**

2.  **Código** (notebook `.ipynb`)
1.  **Relatório** (`.pdf`).

- O PDF deve conter:
  - **Setup** (parâmetros usados).
  - **Resultados** (figuras e tabelas organizadas por experimento).
  - **Análises curtas** por experimento.
- O PDF **NÃO** deve conter:

  - Códigos.

  ***

# iteracao de politica truncada ( usa o gymnaisum )

1.  Implemente o algoritmo **iteração de política truncada**.
2.  Gere um **gráfico de dispersão** em que cada ponto (x,y) corresponde à (valor do j_truncado, iteração em que a condição de convergência foi satisfeita para este j_truncado).

** Utilize a seguinte configuração do ambiente FrozenLake para os experimentos**

- `map_name = '8x8'` e `map_name = '4x4'`
- `render_mode="rgb_array"`
- `is_slippery=True`

**No experimento com configuração `map_name = '4x4'` mostrar:**

1.  **Figuras**:
    - heatmap de $V(s)$ (função `plot_tabular`);
    - heatmap de $Q(s,a)$ (função `plot_tabular`);
    - heatmap de $\pi(a\mid s)$ (função `plot_tabular`);
    - gráficos de barras de $\pi(a\mid s)$ (função `visualizar_politica`).
    - gráfico de dispersão

**No experimento com configuração `map_name = '8x8'` mostrar:**

1.  **Figura**:
    - gráfico de dispersão

**Entregáveis:**

2.  **Código** (notebook `.ipynb`)
1.  **Relatório** (`.pdf`).

- O PDF deve conter:
  - **Setup** (parâmetros usados).
  - **Resultados** (figuras e tabelas organizadas por experimento).
  - **Análises curtas** por experimento.
- O PDF **NÃO** deve conter:
  - Códigos.

# mc basico


1.  Implemente o algoritmo **MC básico**.

2.  Analise o impacto o comprimento do episódio (`T`):

- Fixe o número de episódios (`N=1`) e o fator de desconto ($\gamma=0.9$).

- Varie o comprimento do episódio (`T` $\in \{1, 5, 10, 15, 30\}$).

2.  Analise o impacto do fator de desconto ($\gamma$):

- Fixe o comprimento do episódio (`T=30`) e o número de episódios (`N=1`).

- Varie o fator de desconto ($\gamma \in \{0.0, 0.5, 0.9, 0.95, 0.99\}$).

**Configuração base (baseline)**

- `world_size = (5, 5)`
- `target_states = [(3, 2)]`
- `bad_states = [(1, 1), (1, 2), (2, 2), (3, 1), (3, 3), (4, 1)]`
- `allow_bad_entry = True`
- recompensas base: $[\,r_{\text{boundary}},\ r_{\text{bad}},\ r_{\text{target}},\ r_{\text{other}}\,] = [-1,\ -10,\ 1,\ 0]$
- `max_iter=20`

**Em todos os experimentos mostrar:**

1.  **Figuras**:
    - heatmap de $V(s)$ (função `plot_tabular`);
    - heatmap de $Q(s,a)$ (função `plot_tabular`);
    - heatmap de $\pi(a\mid s)$ (função `plot_tabular`);
    - politica aprendida (função `plot_policy`)
2.  **Discussão**: texto breve (3-6 linhas) por experimento.

**Entregáveis:**

2.  **Código** (notebook `.ipynb`)
1.  **Relatório** (`.pdf`).

- O PDF deve conter:
  - **Setup** (parâmetros usados).
  - **Resultados** (figuras e tabelas organizadas por experimento).
  - **Análises curtas** por experimento.
- O PDF **NÃO** deve conter:
  - Códigos.

# mc inicios exploratorios

1.  Implemente o algoritmo **MC com inícios exploratórios**.

2.  Analise o impacto do número de episódios (`N`):

- Fixe o comprimento do episódio (`T=100`) e o fator de desconto ($\gamma=0.9$).

- Varie ($N \in \{10, 100, 1000, 10000\}$).

3.  Analise o impacto o comprimento do episódio (`T`):

- Fixe o número de episódios (`N=10000`) e o fator de desconto ($\gamma=0.9$).

- Varie o comprimento do episódio (`T` $\in \{1, 10, 50, 100\}$).

4.  Analise o impacto do fator de desconto ($\gamma$):

- Fixe o comprimento do episódio (`T=100`) e o número de episódios (`N=10000`).

- Varie o fator de desconto ($\gamma \in \{0.0, 0.5, 0.9, 0.95, 0.99\}$).

**Configuração base:**

- `world_size = (5, 5)`
- `target_states = [(3, 2)]`
- `bad_states = [(1, 1), (1, 2), (2, 2), (3, 1), (3, 3), (4, 1)]`
- `allow_bad_entry = True`
- recompensas base: $[\,r_{\text{boundary}},\ r_{\text{bad}},\ r_{\text{target}},\ r_{\text{other}}\,] = [-1,\ -10,\ 1,\ 0]$
- `max_iter=20`

**Em todos os experimentos mostrar:**

1.  **Figuras**:
    - heatmap de $V(s)$ (função `plot_tabular`);
    - heatmap de $Q(s,a)$ (função `plot_tabular`);
    - heatmap de $\pi(a\mid s)$ (função `plot_tabular`);
    - política aprendida (função `plot_policy`)
    - número de visitas por par (s,a) (função `plot_visitas_log` ou `plot_tabular`)
2.  **Discussão**: texto breve (3-6 linhas) por experimento.

**Entregáveis:**

2.  **Código** (notebook `.ipynb`)
1.  **Relatório** (`.pdf`).

- O PDF deve conter:
  - **Setup** (parâmetros usados).
  - **Resultados** (figuras e tabelas organizadas por experimento).
  - **Análises curtas** por experimento.
- O PDF **NÃO** deve conter:
  - Códigos.

# mc e greedy


1.  Implemente o algoritmo **MC $\epsilon$-guloso**.

2.  Analise o impacto do número de episódios ($N$):

    - Fixe o comprimento do episódio ($T=10^5$), o fator de desconto ($\gamma=0.9$) e o parâmetro $\epsilon$ ($\epsilon=0.5$).

    - Varie ($N \in \{2, 10, 20, 50\}$).

3.  Analise o impacto do comprimento do episódio ($T$):

    - Fixe o número de episódios ($N=20$) e o fator de desconto ($\gamma=0.9$) e o parâmetro $\epsilon$ ($\epsilon=0.5$).

    - Varie o comprimento do episódio ($T \in \{10^3, 10^4, 10^5, 10^6\}$).

4.  Analise o impacto do parâmetro $\epsilon$ :

    - Fixe o comprimento do episódio ($T=10^5$), o número de episódios ($N=20$) e o fator de desconto ($\gamma=0.9$).

    - Varie o parâmetro $\epsilon$ ($\epsilon \in \{0.0, 0.25, 0.5, 0.75, 1.0\}$).

5.  Compare o desempenho do algoritmo **MC $\epsilon$-guloso** com **MC com inícios exploratórios**.

    - Configuração para ambos os algoritmos: $N=20$, $T=10^5$, $\gamma=0.9$.

    - Para o **MC $\epsilon$-guloso** utilize $\epsilon=0.5$.

**Configuração base:**

- `world_size = (5, 5)`
- `target_states = [(3, 2)]`
- `bad_states = [(1, 1), (1, 2), (2, 2), (3, 1), (3, 3), (4, 1)]`
- `allow_bad_entry = True`
- recompensas base: $[\,r_{\text{boundary}},\ r_{\text{bad}},\ r_{\text{target}},\ r_{\text{other}}\,] = [-1,\ -1,\ 1,\ 0]$

**Em todos os experimentos mostrar:**

1.  **Figuras**:

    - heatmap de $V(s)$ (função `plot_tabular`);
    - heatmap de $Q(s,a)$ (função `plot_tabular`);
    - heatmap de $\pi(a\mid s)$ (função `plot_tabular`);
    - política aprendida (função `plot_policy`)
    - número de visitas por par (s,a) (função `plot_visitas_log` ou `plot_tabular`)

2.  **Discussão**: texto breve (3-6 linhas) por experimento.

**Entregáveis:**

2.  **Código** (notebook `*.ipynb`)
1.  **Relatório** (`*.pdf`).

- O PDF deve conter:
  - **Setup** (parâmetros usados).
  - **Resultados** (figuras e tabelas organizadas por experimento).
  - **Análises curtas** por experimento.
- O PDF **NÃO** deve conter:
  - Códigos.
