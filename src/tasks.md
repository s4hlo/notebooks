<style>
h1 { color: #cba6f7; } /* Pink */
h2 { color: #89b4fa; } /* Blue */
h3 { color: #a6e3a1; } /* Green */
h4 { color: #fab387; } /* Peach */
</style>

# Tarefas de Aprendizado por Reforço

## Índice

1. [Instruções Gerais](#instruções-gerais)
2. [Setup Base](#setup-base)
3. [Laboratórios](#laboratórios)
   - [3.1 Iteração de Valor](#31-iteração-de-valor)
   - [3.2 Iteração de Política](#32-iteração-de-política)
   - [3.3 Iteração de Política Truncada](#33-iteração-de-política-truncada)
   - [3.4 MC Básico](#34-mc-básico)
   - [3.5 MC Inícios Exploratórios](#35-mc-inícios-exploratórios)
   - [3.6 MC Epsilon Greedy](#36-mc-epsilon-greedy)

---

## Instruções Gerais

Em cada laboratório, implemente o algoritmo correspondente e realize os testes descritos.

> **Importante**: Se alterar qualquer parâmetro do setup, **documente explicitamente** no relatório.

### Entregáveis

1. **Código** (notebook `*.ipynb`)
2. **Relatório** (`*.pdf`)

#### Conteúdo do PDF:
- **Setup** (parâmetros usados)
- **Resultados** (figuras e tabelas organizadas por experimento)
- **Análises curtas** por experimento

#### O PDF **NÃO** deve conter:
- Códigos

---

## Setup Base

### Configuração do Gridworld

```python
world_size = (5, 5)
target_states = [(3, 2)]
bad_states = [(1, 1), (1, 2), (2, 2), (3, 1), (3, 3), (4, 1)]
allow_bad_entry = True
```

> **Nota**: As configurações específicas de cada algoritmo (recompensas, parâmetros de convergência, etc.) são detalhadas nas seções individuais abaixo.

---

## Laboratórios

### 3.1 Iteração de Valor

#### Configuração Específica

> - recompensas base: $[\,r_{\text{boundary}},\ r_{\text{bad}},\ r_{\text{target}},\ r_{\text{other}}\,] = [-1,\ -10,\ 1,\ 0]$
> - tolerância e limite: $\theta = 10^{-6}$, `max_iterações = 1000`

#### Experimentos

1. **Variação do fator de desconto**
   - Observar e reportar o efeito de diferentes valores da taxa de desconto (por exemplo: $\gamma \in \{\,0.0,\ 0.5,\ 0.9\,\}$)

2. **Penalidade de estados ruins mais branda**
   - Observar e reportar o efeito de trocar $r_{\text{bad}}=-10$ para $r_{\text{bad}}=-1$.

3. **Transformação afim nas recompensas**
   - Observar e reportar o efeito de uma transformação afim ($r' = a\,r + b$, com $a>0$) em todas as recompensas, isto é, em todos os elementos de $[\,r_{\text{boundary}}, r_{\text{bad}}, r_{\text{target}}, r_{\text{other}}\,]$.

#### Resultados Esperados

**Em todos os experimentos mostrar:**

1. **Figuras**:
   - heatmap de $V(s)$ no grid $(n_{\text{rows}}\times n_{\text{cols}})$;
   - heatmap de $Q(s,a)$ (ações nas linhas, estados nas colunas);
   - heatmap de $\pi(a\mid s)$ (probabilidades).
2. **Convergência**: número de iterações até $\lVert v_{k+1}-v_k\rVert_\infty < \theta$.
3. **Discussão**: texto breve (3–6 linhas) por experimento.

### 3.2 Iteração de Política

#### Configuração Específica

> - recompensas base: $[\,r_{\text{boundary}},\ r_{\text{bad}},\ r_{\text{target}},\ r_{\text{other}}\,] = [-1,\ -10,\ 1,\ 0]$
> - tolerância e limite: $\theta = 10^{-6}$, `max_iterações = 1000`

#### Experimentos

1. **Comparação de algoritmos**
   - Compare os algoritmos de iteração de valor (laboratório 1) e de iteração de política (laboratório 2) quanto ao número de iterações utilizadas até a condição de convergência ser satisfeita.

#### Resultados Esperados

**Em todos os experimentos mostrar:**

1. **Figuras**:
   - heatmap de $V(s)$ no grid $(n_{\text{rows}}\times n_{\text{cols}})$;
   - heatmap de $Q(s,a)$ (ações nas linhas, estados nas colunas);
   - heatmap de $\pi(a\mid s)$ (probabilidades).
2. **Convergência**: número de iterações até $\lVert v_{\pi_k}-v_{\pi_{k-1}}\rVert_\infty < \theta_{política}$.
3. **Discussão**: texto breve (3-6 linhas) por experimento.

### 3.3 Iteração de Política Truncada

> **Nota**: Este é o único laboratório que não usa o gridworld

#### Configuração do Ambiente FrozenLake

```python
map_name = '8x8'  # e '4x4' SAHLO FOLINAAAAA
render_mode = "rgb_array"
is_slippery = True
```

#### Experimentos

1. **Gráfico de dispersão**
   - Gere um gráfico de dispersão em que cada ponto (x,y) corresponde à (valor do j_truncado, iteração em que a condição de convergência foi satisfeita para este j_truncado).

#### Resultados Esperados

##### Configuração `map_name = '4x4'`

1. **Figuras**:
   - heatmap de $V(s)$ (função `plot_tabular`)
   - heatmap de $Q(s,a)$ (função `plot_tabular`)
   - heatmap de $\pi(a\mid s)$ (função `plot_tabular`)
   - gráficos de barras de $\pi(a\mid s)$ (função `visualizar_politica`)
   - gráfico de dispersão

##### Configuração `map_name = '8x8'`

1. **Figura**:
   - gráfico de dispersão

### 3.4 MC Básico

#### Configuração Específica

> - recompensas base: $[\,r_{\text{boundary}},\ r_{\text{bad}},\ r_{\text{target}},\ r_{\text{other}}\,] = [-1,\ -10,\ 1,\ 0]$
> - `max_iter=20`

#### Experimentos

1. **Análise do comprimento do episódio (`T`)**
   - Fixe o número de episódios (`N=1`) e o fator de desconto ($\gamma=0.9$)
   - Varie o comprimento do episódio (`T` $\in \{1, 5, 10, 15, 30\}$)

2. **Análise do fator de desconto ($\gamma$)**
   - Fixe o comprimento do episódio (`T=30`) e o número de episódios (`N=1`)
   - Varie o fator de desconto ($\gamma \in \{0.0, 0.5, 0.9, 0.95, 0.99\}$)

#### Resultados Esperados

**Em todos os experimentos mostrar:**

1. **Figuras**:
   - heatmap de $V(s)$ (função `plot_tabular`)
   - heatmap de $Q(s,a)$ (função `plot_tabular`)
   - heatmap de $\pi(a\mid s)$ (função `plot_tabular`)
   - política aprendida (função `plot_policy`)
2. **Discussão**: texto breve (3-6 linhas) por experimento.

### 3.5 MC Inícios Exploratórios

#### Configuração Específica

> - recompensas base: $[\,r_{\text{boundary}},\ r_{\text{bad}},\ r_{\text{target}},\ r_{\text{other}}\,] = [-1,\ -10,\ 1,\ 0]$
> - `max_iter=20`

#### Experimentos

1. **Análise do número de episódios (`N`)**
   - Fixe o comprimento do episódio (`T=100`) e o fator de desconto ($\gamma=0.9$)
   - Varie ($N \in \{10, 100, 1000, 10000\}$)

2. **Análise do comprimento do episódio (`T`)**
   - Fixe o número de episódios (`N=10000`) e o fator de desconto ($\gamma=0.9$)
   - Varie o comprimento do episódio (`T` $\in \{1, 10, 50, 100\}$)

3. **Análise do fator de desconto ($\gamma$)**
   - Fixe o comprimento do episódio (`T=100`) e o número de episódios (`N=10000`)
   - Varie o fator de desconto ($\gamma \in \{0.0, 0.5, 0.9, 0.95, 0.99\}$)

#### Resultados Esperados

**Em todos os experimentos mostrar:**

1. **Figuras**:
   - heatmap de $V(s)$ (função `plot_tabular`)
   - heatmap de $Q(s,a)$ (função `plot_tabular`)
   - heatmap de $\pi(a\mid s)$ (função `plot_tabular`)
   - política aprendida (função `plot_policy`)
   - número de visitas por par (s,a) (função `plot_visitas_log` ou `plot_tabular`)
2. **Discussão**: texto breve (3-6 linhas) por experimento.

### 3.6 MC Epsilon Greedy

#### Configuração Específica

> - recompensas base: $[\,r_{\text{boundary}},\ r_{\text{bad}},\ r_{\text{target}},\ r_{\text{other}}\,] = [-1,\ -1,\ 1,\ 0]$

#### Experimentos

1. **Análise do número de episódios ($N$)**
   - Fixe o comprimento do episódio ($T=10^5$), o fator de desconto ($\gamma=0.9$) e o parâmetro $\epsilon$ ($\epsilon=0.5$)
   - Varie ($N \in \{2, 10, 20, 50\}$)

2. **Análise do comprimento do episódio ($T$)**
   - Fixe o número de episódios ($N=20$) e o fator de desconto ($\gamma=0.9$) e o parâmetro $\epsilon$ ($\epsilon=0.5$)
   - Varie o comprimento do episódio ($T \in \{10^3, 10^4, 10^5, 10^6\}$)

3. **Análise do parâmetro $\epsilon$**
   - Fixe o comprimento do episódio ($T=10^5$), o número de episódios ($N=20$) e o fator de desconto ($\gamma=0.9$)
   - Varie o parâmetro $\epsilon$ ($\epsilon \in \{0.0, 0.25, 0.5, 0.75, 1.0\}$)

4. **Comparação de algoritmos**
   - Compare o desempenho do algoritmo **MC $\epsilon$-guloso** com **MC com inícios exploratórios**
   - Configuração para ambos os algoritmos: $N=20$, $T=10^5$, $\gamma=0.9$
   - Para o **MC $\epsilon$-guloso** utilize $\epsilon=0.5$

#### Resultados Esperados

**Em todos os experimentos mostrar:**

1. **Figuras**:
   - heatmap de $V(s)$ (função `plot_tabular`)
   - heatmap de $Q(s,a)$ (função `plot_tabular`)
   - heatmap de $\pi(a\mid s)$ (função `plot_tabular`)
   - política aprendida (função `plot_policy`)
   - número de visitas por par (s,a) (função `plot_visitas_log` ou `plot_tabular`)

2. **Discussão**: texto breve (3-6 linhas) por experimento.
