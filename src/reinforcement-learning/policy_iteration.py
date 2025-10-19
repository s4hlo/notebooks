# %%
import numpy as np
from typing import Tuple
from utils.ambiente import AmbienteNavegacaoLabirinto
from utils.plot import plot_policy, plot_tabular


# %%
def iteracao_de_politica(
    ambiente: "AmbienteNavegacaoLabirinto",
    gamma: float = 0.9,
    theta_avaliacao: float = 1e-6,  # tolerância do loop interno (avaliar_politica)
    theta_politica: float = 1e-6,  # tolerância do loop externo (||V - V_old||_inf)
    max_iteracoes_politica: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Iteração de Política.

    Executa ciclos de:
    1) avaliação da política até ||V_pi_k^(j) − V_pi_k^(j-1)||_inf < theta_avaliacao;
    2) melhoria gulosa da política em relação a Q.
    Para quando ||V_pi_k − V_pi_{k-1}||_inf < theta_politica ou ao atingir max_iteracoes_politica.

    Parâmetros
    ----------
    ambiente : AmbienteNavegacaoLabirinto
        Ambiente tabular com P(r|s,a) e P(s'|s,a).
    gamma : float, padrão=0.9
        Fator de desconto (0 ≤ gamma < 1).
    theta_avaliacao : float, padrão=1e-6
        Tolerância do loop interno (||V_pi_k^(j) − V_pi_k^(j-1)||_inf).
    theta_politica : float, padrão=1e-6
        Tolerância do loop externo (||V_pi_k − V_pi_{k-1}||_inf).
    max_iteracoes_politica : int, padrão=1000
        Limite máximo de ciclos (avaliação + melhoria).

    Retorna
    -------
    V : np.ndarray, shape (n_estados,)
        Valores de estado.
    Q : np.ndarray, shape (n_estados, n_acoes)
        Valores de ação.
    Pi : np.ndarray, shape (n_estados, n_acoes)
        Política determinística.
    num_iter : int
        Número de ciclos externos executados.
    """
    
    print(f"Iniciando Iteração de Política com γ={gamma}, θ_avaliação={theta_avaliacao}, θ_política={theta_politica}")
    
    # Atalhos (utilize as variáveis que forem necessárias)
    n_estados = ambiente.n_states
    n_acoes = ambiente.n_actions
    n_recompensas = ambiente.n_rewards
    R = ambiente.recompensas_imediatas  # r(s,a) -> r
    T = ambiente.transicao_de_estados  # T(s,a) -> s'
    Ps = (
        ambiente.state_transition_probabilities
    )  # shape (n_estados, n_estados, n_acoes)
    Pr = ambiente.reward_probabilities  # shape (n_recompensas,  n_estados, n_acoes)
    r_vector = ambiente.recompensas_possiveis  # shape (n_recompensas,)

    # Inicialização: V, Q, Pi
    V = np.zeros(n_estados)  # Valores de estado
    Q = np.zeros((n_estados, n_acoes))  # Valores de ação
    Pi = np.ones((n_estados, n_acoes)) / n_acoes  # Política uniforme inicial
    
    print(f"Inicialização: {n_estados} estados, {n_acoes} ações")
    print(f"Política inicial: uniforme (probabilidade 1/{n_acoes} para cada ação)")

    for k in range(max_iteracoes_politica):
        print(f"\n--- Iteração Externa {k+1} ---")
        
        # 1. AVALIAÇÃO DA POLÍTICA (loop interno)
        print("1. Avaliando política atual...")
        V_old = V.copy()
        
        # Iteração até convergência dos valores de estado
        for j in range(1000):  # Loop interno para avaliação da política
            V_prev = V.copy()
            
            for s in range(n_estados):
                # Calcular valor do estado s sob a política atual
                V[s] = 0.0
                for a in range(n_acoes):
                    # Recompensa esperada imediata
                    R_sa = 0.0
                    for r_idx in range(n_recompensas):
                        R_sa += Pr[r_idx, s, a] * r_vector[r_idx]
                    
                    # Valor futuro esperado
                    future_value = 0.0
                    for s_next in range(n_estados):
                        future_value += Ps[s_next, s, a] * V_prev[s_next]
                    
                    # Contribuição da ação a para o valor do estado s
                    V[s] += Pi[s, a] * (R_sa + gamma * future_value)
            
            # Teste de convergência do loop interno
            delta_inner = np.max(np.abs(V - V_prev))
            if delta_inner < theta_avaliacao:
                print(f"   Convergência da avaliação em {j+1} iterações (δ={delta_inner:.2e})")
                break
        
        # 2. MELHORIA DA POLÍTICA
        print("2. Melhorando política...")
        Pi_old = Pi.copy()
        
        # Calcular Q(s,a) para todos os pares (s,a)
        for s in range(n_estados):
            for a in range(n_acoes):
                # Recompensa esperada imediata
                R_sa = 0.0
                for r_idx in range(n_recompensas):
                    R_sa += Pr[r_idx, s, a] * r_vector[r_idx]
                
                # Valor futuro esperado
                future_value = 0.0
                for s_next in range(n_estados):
                    future_value += Ps[s_next, s, a] * V[s_next]
                
                Q[s, a] = R_sa + gamma * future_value
        
        # Política gulosa: escolher ação com maior Q(s,a)
        for s in range(n_estados):
            best_action = np.argmax(Q[s, :])
            Pi[s, :] = 0.0
            Pi[s, best_action] = 1.0
        
        # 3. TESTE DE CONVERGÊNCIA DO LOOP EXTERNO
        delta_outer = np.max(np.abs(V - V_old))
        policy_stable = np.array_equal(Pi, Pi_old)
        
        print(f"   δ_valores = {delta_outer:.2e}, política estável = {policy_stable}")
        
        if delta_outer < theta_politica or policy_stable:
            print(f"Convergência atingida em {k+1} iterações externas!")
            if policy_stable:
                print("   Motivo: política não mudou")
            else:
                print(f"   Motivo: δ_valores < θ_política ({delta_outer:.2e} < {theta_politica})")
            break
    
    print(f"\nResultado final: {k+1} iterações externas executadas")
    return V, Q, Pi, k + 1


# %% [markdown]
# # Laboratório 3.2: Iteração de Política
# 
# Este laboratório implementa o algoritmo de **Iteração de Política** para resolver um problema de navegação em labirinto.
# 
# ## Configuração Base
# 
# - **Tamanho do mundo**: 5x5
# - **Estados ruins**: (1,1), (1,2), (2,2), (3,1), (3,3), (4,1)
# - **Estado alvo**: (3,2)
# - **Recompensas base**: [boundary, bad, target, other] = [-1, -10, 1, 0]
# - **Parâmetros de convergência**: θ_avaliação = 10⁻⁶, θ_política = 10⁻⁶, max_iterações = 1000
# 
# ## Algoritmo de Iteração de Política
# 
# O algoritmo executa ciclos de:
# 1. **Avaliação da política**: Calcula V^π(s) para todos os estados até convergência
# 2. **Melhoria da política**: Atualiza π(s) = argmax_a Q(s,a) onde Q(s,a) = R(s,a) + γ∑P(s'|s,a)V(s')
# 
# Para quando a política não muda mais ou quando ||V_k - V_{k-1}||_∞ < θ_política

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
# ## Experimento 1: Comparação de Algoritmos
# 
# **Objetivo**: Comparar os algoritmos de iteração de valor (laboratório 1) e de iteração de política (laboratório 2) quanto ao número de iterações utilizadas até a condição de convergência ser satisfeita.
# 
# **Hipótese**: A iteração de política geralmente converge em menos iterações externas que a iteração de valor, pois cada iteração externa inclui uma avaliação completa da política.

# %%
print("=== EXECUTANDO ITERAÇÃO DE POLÍTICA ===")
print("Parâmetros: γ=0.9, θ_avaliação=1e-6, θ_política=1e-6")

V, Q, Pi, k = iteracao_de_politica(
    ambiente,  # gridworld
    gamma=0.9,  # fator de desconto (0 <= gamma < 1)
    theta_avaliacao=1e-6,  # convergência do loop interno
    theta_politica=1e-6,  # convergência do loop externo
)

print(f"\n=== RESULTADOS ===")
print(f"Número de iterações externas: {k}")
print(f"Valores de estado (V) - min: {V.min():.4f}, max: {V.max():.4f}")
print(f"Valores de ação (Q) - min: {Q.min():.4f}, max: {Q.max():.4f}")

# %% [markdown]
# ### Análise dos Resultados
# 
# **Convergência**: O algoritmo convergiu em X iterações externas.
# 
# **Política aprendida**: A política determinística resultante indica a melhor ação para cada estado.

# %% [markdown]
# ### Visualização dos Resultados

# %%
print("=== VISUALIZAÇÕES ===")
print("1. Heatmap dos valores de ação Q(s,a)")
plot_tabular(Q, kind="Q")

print("\n2. Heatmap da política π(a|s)")
plot_tabular(Pi, kind="Pi")

print("\n3. Heatmap dos valores de estado V(s)")
plot_tabular(V, kind="V", ambiente=ambiente, center_zero=False)

print("\n4. Política aprendida no labirinto")
_ = plot_policy(ambiente, Pi)

# %% [markdown]
# ## Comparação com Iteração de Valor
# 
# Para comparar a eficiência dos algoritmos, vamos executar também a iteração de valor com os mesmos parâmetros.

# %%
print("=== COMPARAÇÃO COM ITERAÇÃO DE VALOR ===")
print("Executando iteração de valor com os mesmos parâmetros...")

# Importar a função de iteração de valor
from value_iteration import iteracao_de_valor

V_vi, Q_vi, Pi_vi, k_vi = iteracao_de_valor(
    ambiente,
    gamma=0.9,
    theta=1e-6,
    max_iteracoes=1000
)

print(f"\n=== COMPARAÇÃO DE RESULTADOS ===")
print(f"Iteração de Política: {k} iterações externas")
print(f"Iteração de Valor: {k_vi} iterações")
print(f"Diferença: {k_vi - k} iterações a mais para iteração de valor")

# Verificar se as políticas são equivalentes
policy_equivalent = np.allclose(Pi, Pi_vi, atol=1e-10)
print(f"Políticas equivalentes: {policy_equivalent}")

if policy_equivalent:
    print("✓ Ambos os algoritmos convergiram para a mesma política ótima")
else:
    print("⚠ As políticas diferem - verificar implementações")

# %% [markdown]
# ### Análise da Comparação
# 
# **Resultados**: A iteração de política convergiu em X iterações externas, enquanto a iteração de valor convergiu em Y iterações.
# 
# **Discussão**: A iteração de política geralmente converge em menos iterações externas que a iteração de valor porque:
# - Cada iteração externa da iteração de política inclui uma avaliação completa da política atual
# - A iteração de valor atualiza simultaneamente valores e política a cada iteração
# - A convergência da política é um critério mais forte que a convergência dos valores

# %%
