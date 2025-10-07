# %%
import numpy as np
from typing import Tuple
from utils.ambiente import AmbienteNavegacaoLabirinto
from utils.plot import plot_policy, plot_tabular


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
# # Laboratório 3.1: Iteração de Valor
# 
# Este laboratório implementa o algoritmo de **Iteração de Valor** para resolver um problema de navegação em labirinto.
# 
# ## Configuração Base
# 
# - **Tamanho do mundo**: 5x5
# - **Estados ruins**: (1,1), (1,2), (2,2), (3,1), (3,3), (4,1)
# - **Estado alvo**: (3,2)
# - **Recompensas base**: [boundary, bad, target, other] = [-1, -10, 1, 0]
# - **Parâmetros de convergência**: θ = 10⁻⁶, max_iterações = 1000

# %%
# Configuração inicial do ambiente
print("=== CONFIGURAÇÃO INICIAL DO AMBIENTE ===")
print("Criando ambiente de navegação 5x5 com estados ruins e alvo...")

ambiente = AmbienteNavegacaoLabirinto(
    world_size=(5, 5),
    bad_states=[(1, 1), (1, 2), (2, 2), (3, 1), (3, 3), (4, 1)],
    target_states=[(3, 2)],
    allow_bad_entry=True,
    rewards=[-1, -10, 1, 0],
)

print(f"Estados totais: {ambiente.n_states}")
print(f"Ações possíveis: {ambiente.n_actions}")
print(f"Recompensas: {ambiente.recompensas_possiveis}")
print("\nVisualização do labirinto:")
ambiente.plot_labirinto()

# %% [markdown]
# ## Experimento 1: Variação do Fator de Desconto
# 
# **Objetivo**: Analisar o efeito de diferentes valores da taxa de desconto γ ∈ {0.0, 0.5, 0.9}
# 
# **Hipótese**: Valores menores de γ fazem o agente focar em recompensas imediatas, enquanto valores maiores consideram recompensas futuras.

# %%
print("=== EXPERIMENTO 1: VARIAÇÃO DO FATOR DE DESCONTO ===")
print("Analisando convergência para diferentes valores de γ...")

gammas = [0.0, 0.5, 0.9]
resultados_gamma = {}

for gamma in gammas:
    print(f"\n--- Executando iteração de valor com γ = {gamma} ---")
    V, Q, Pi, iteracoes = iteracao_de_valor(ambiente, gamma=gamma, theta=1e-6, max_iteracoes=1000)
    
    print(f"Convergiu em {iteracoes} iterações")
    print(f"Valor máximo V(s): {np.max(V):.6f}")
    print(f"Valor mínimo V(s): {np.min(V):.6f}")
    
    resultados_gamma[gamma] = {
        'V': V, 'Q': Q, 'Pi': Pi, 'iteracoes': iteracoes
    }

# %% [markdown]
# ### Resultados - Variação do Fator de Desconto

# %%
print("\n=== RESULTADOS EXPERIMENTO 1 ===")
print("Comparando convergência para diferentes valores de γ:")

for gamma in gammas:
    iteracoes = resultados_gamma[gamma]['iteracoes']
    V = resultados_gamma[gamma]['V']
    print(f"γ = {gamma:3.1f}: {iteracoes:3d} iterações, V_max = {np.max(V):7.4f}, V_min = {np.min(V):7.4f}")

# %%
# Visualizações para γ = 0.9 (caso base)
print("\n=== VISUALIZAÇÕES PARA γ = 0.9 ===")
V_base, Q_base, Pi_base = resultados_gamma[0.9]['V'], resultados_gamma[0.9]['Q'], resultados_gamma[0.9]['Pi']

print("1. Heatmap da função valor V(s):")
plot_tabular(V_base, kind="V", ambiente=ambiente, center_zero=False)

print("\n2. Heatmap da função ação-valor Q(s,a):")
plot_tabular(Q_base, kind="Q")

print("\n3. Heatmap da política π(a|s):")
plot_tabular(Pi_base, kind="Pi")

print("\n4. Política aprendida (setas):")
_ = plot_policy(ambiente, Pi_base)

# %% [markdown]
# ### Análise - Variação do Fator de Desconto
# 
# **Observações**:
# - **γ = 0.0**: O agente ignora completamente recompensas futuras, focando apenas na recompensa imediata
# - **γ = 0.5**: Balance entre recompensas imediatas e futuras, convergência moderada
# - **γ = 0.9**: Maior consideração de recompensas futuras, convergência mais lenta mas política mais sofisticada
# 
# **Convergência**: Valores menores de γ convergem mais rapidamente, mas produzem políticas menos otimizadas.

# %% [markdown]
# ## Experimento 2: Penalidade de Estados Ruins Mais Branda
# 
# **Objetivo**: Analisar o efeito de trocar r_bad de -10 para -1
# 
# **Hipótese**: Penalidade mais branda deve resultar em políticas menos aversivas aos estados ruins.

# %%
print("=== EXPERIMENTO 2: PENALIDADE BRANDA DE ESTADOS RUINS ===")
print("Comparando r_bad = -10 vs r_bad = -1...")

# Ambiente com penalidade severa (original)
print("\n--- Configuração original: r_bad = -10 ---")
ambiente_severo = AmbienteNavegacaoLabirinto(
    world_size=(5, 5),
    bad_states=[(1, 1), (1, 2), (2, 2), (3, 1), (3, 3), (4, 1)],
    target_states=[(3, 2)],
    allow_bad_entry=True,
    rewards=[-1, -10, 1, 0],  # r_bad = -10
)

V_severo, Q_severo, Pi_severo, iter_severo = iteracao_de_valor(
    ambiente_severo, gamma=0.9, theta=1e-6, max_iteracoes=1000
)
print(f"Convergiu em {iter_severo} iterações")

# Ambiente com penalidade branda
print("\n--- Configuração branda: r_bad = -1 ---")
ambiente_brando = AmbienteNavegacaoLabirinto(
    world_size=(5, 5),
    bad_states=[(1, 1), (1, 2), (2, 2), (3, 1), (3, 3), (4, 1)],
    target_states=[(3, 2)],
    allow_bad_entry=True,
    rewards=[-1, -1, 1, 0],  # r_bad = -1
)

V_brando, Q_brando, Pi_brando, iter_brando = iteracao_de_valor(
    ambiente_brando, gamma=0.9, theta=1e-6, max_iteracoes=1000
)
print(f"Convergiu em {iter_brando} iterações")

# %% [markdown]
# ### Resultados - Penalidade Branda

# %%
print("\n=== RESULTADOS EXPERIMENTO 2 ===")
print("Comparando políticas com diferentes penalidades:")

print(f"\nPenalidade severa (r_bad = -10):")
print(f"  Iterações: {iter_severo}")
print(f"  V_max: {np.max(V_severo):.4f}, V_min: {np.min(V_severo):.4f}")

print(f"\nPenalidade branda (r_bad = -1):")
print(f"  Iterações: {iter_brando}")
print(f"  V_max: {np.max(V_brando):.4f}, V_min: {np.min(V_brando):.4f}")

# %%
print("\n=== VISUALIZAÇÕES EXPERIMENTO 2 ===")

print("1. Política com penalidade SEVERA (r_bad = -10):")
plot_tabular(V_severo, kind="V", ambiente=ambiente_severo, center_zero=False)
plot_tabular(Pi_severo, kind="Pi")
_ = plot_policy(ambiente_severo, Pi_severo)

print("\n2. Política com penalidade BRANDA (r_bad = -1):")
plot_tabular(V_brando, kind="V", ambiente=ambiente_brando, center_zero=False)
plot_tabular(Pi_brando, kind="Pi")
_ = plot_policy(ambiente_brando, Pi_brando)

# %% [markdown]
# ### Análise - Penalidade Branda
# 
# **Observações**:
# - **Penalidade severa (-10)**: O agente evita completamente estados ruins, criando rotas mais longas mas seguras
# - **Penalidade branda (-1)**: O agente pode considerar passar por estados ruins se isso levar a uma rota mais curta
# - **Convergência**: Ambas convergem rapidamente, mas com políticas fundamentalmente diferentes
# 
# **Implicações**: A escolha da penalidade afeta diretamente o comportamento exploratório vs conservador do agente.

# %% [markdown]
# ## Experimento 3: Transformação Afim nas Recompensas
# 
# **Objetivo**: Analisar o efeito de uma transformação afim r' = a·r + b (com a > 0) em todas as recompensas
# 
# **Hipótese**: Transformações afins positivas preservam a ordem das recompensas, mantendo a política ótima.

# %%
print("=== EXPERIMENTO 3: TRANSFORMAÇÃO AFIM NAS RECOMPENSAS ===")
print("Aplicando transformação r' = 2r + 1 (a=2, b=1)...")

# Recompensas originais: [-1, -10, 1, 0]
recompensas_originais = np.array([-1, -10, 1, 0])
print(f"Recompensas originais: {recompensas_originais}")

# Transformação afim: r' = 2r + 1
a, b = 2.0, 1.0
recompensas_transformadas = a * recompensas_originais + b
print(f"Recompensas transformadas: {recompensas_transformadas}")

# Ambiente original
print("\n--- Ambiente com recompensas originais ---")
ambiente_original = AmbienteNavegacaoLabirinto(
    world_size=(5, 5),
    bad_states=[(1, 1), (1, 2), (2, 2), (3, 1), (3, 3), (4, 1)],
    target_states=[(3, 2)],
    allow_bad_entry=True,
    rewards=recompensas_originais,
)

V_orig, Q_orig, Pi_orig, iter_orig = iteracao_de_valor(
    ambiente_original, gamma=0.9, theta=1e-6, max_iteracoes=1000
)
print(f"Convergiu em {iter_orig} iterações")

# Ambiente com transformação afim
print("\n--- Ambiente com recompensas transformadas ---")
ambiente_transformado = AmbienteNavegacaoLabirinto(
    world_size=(5, 5),
    bad_states=[(1, 1), (1, 2), (2, 2), (3, 1), (3, 3), (4, 1)],
    target_states=[(3, 2)],
    allow_bad_entry=True,
    rewards=recompensas_transformadas,
)

V_transf, Q_transf, Pi_transf, iter_transf = iteracao_de_valor(
    ambiente_transformado, gamma=0.9, theta=1e-6, max_iteracoes=1000
)
print(f"Convergiu em {iter_transf} iterações")

# %% [markdown]
# ### Resultados - Transformação Afim

# %%
print("\n=== RESULTADOS EXPERIMENTO 3 ===")
print("Comparando políticas com recompensas originais vs transformadas:")

print(f"\nRecompensas originais:")
print(f"  Iterações: {iter_orig}")
print(f"  V_max: {np.max(V_orig):.4f}, V_min: {np.min(V_orig):.4f}")

print(f"\nRecompensas transformadas (r' = 2r + 1):")
print(f"  Iterações: {iter_transf}")
print(f"  V_max: {np.max(V_transf):.4f}, V_min: {np.min(V_transf):.4f}")

# Verificar se as políticas são idênticas
politicas_identicas = np.allclose(Pi_orig, Pi_transf, atol=1e-10)
print(f"\nPolíticas idênticas: {politicas_identicas}")

# %%
print("\n=== VISUALIZAÇÕES EXPERIMENTO 3 ===")

print("1. Política com recompensas ORIGINAIS:")
plot_tabular(V_orig, kind="V", ambiente=ambiente_original, center_zero=False)
plot_tabular(Pi_orig, kind="Pi")
_ = plot_policy(ambiente_original, Pi_orig)

print("\n2. Política com recompensas TRANSFORMADAS:")
plot_tabular(V_transf, kind="V", ambiente=ambiente_transformado, center_zero=False)
plot_tabular(Pi_transf, kind="Pi")
_ = plot_policy(ambiente_transformado, Pi_transf)

# %% [markdown]
# ### Análise - Transformação Afim
# 
# **Observações**:
# - **Políticas idênticas**: A transformação afim preserva a política ótima, como esperado teoricamente
# - **Valores escalados**: Os valores V(s) são transformados pela mesma regra afim
# - **Convergência**: Ambas convergem no mesmo número de iterações
# 
# **Implicações**: Transformações afins positivas são invariantes para problemas de otimização, mantendo a estrutura da solução ótima.

# %% [markdown]
# # Resumo dos Resultados
# 
# ## Tabela de Convergência

# %%
# Criar tabela de resultados
import pandas as pd

dados_tabela = [
    ["1.1", "γ = 0.0", resultados_gamma[0.0]['iteracoes'], 
     f"{np.max(resultados_gamma[0.0]['V']):.4f}", f"{np.min(resultados_gamma[0.0]['V']):.4f}"],
    ["1.2", "γ = 0.5", resultados_gamma[0.5]['iteracoes'], 
     f"{np.max(resultados_gamma[0.5]['V']):.4f}", f"{np.min(resultados_gamma[0.5]['V']):.4f}"],
    ["1.3", "γ = 0.9", resultados_gamma[0.9]['iteracoes'], 
     f"{np.max(resultados_gamma[0.9]['V']):.4f}", f"{np.min(resultados_gamma[0.9]['V']):.4f}"],
    ["2.1", "r_bad = -10", iter_severo, 
     f"{np.max(V_severo):.4f}", f"{np.min(V_severo):.4f}"],
    ["2.2", "r_bad = -1", iter_brando, 
     f"{np.max(V_brando):.4f}", f"{np.min(V_brando):.4f}"],
    ["3.1", "Recompensas originais", iter_orig, 
     f"{np.max(V_orig):.4f}", f"{np.min(V_orig):.4f}"],
    ["3.2", "Recompensas transformadas", iter_transf, 
     f"{np.max(V_transf):.4f}", f"{np.min(V_transf):.4f}"]
]

df_resultados = pd.DataFrame(dados_tabela, 
                           columns=['Experimento', 'Configuração', 'Iterações', 'V_max', 'V_min'])

df_resultados

# 
# ## Conclusões
# 
# 1. **Fator de desconto**: Valores menores convergem mais rapidamente mas produzem políticas menos sofisticadas
# 2. **Penalidades**: Ajustar penalidades de estados ruins afeta diretamente o comportamento exploratório do agente
# 3. **Transformações afins**: Preservam a política ótima, confirmando propriedades teóricas da programação dinâmica

# %%
