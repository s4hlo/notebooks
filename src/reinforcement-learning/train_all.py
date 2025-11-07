"""
Script para treinar todos os algoritmos de RL: SARSA, Expected SARSA, N-step SARSA e Q-learning.
Usa o ambiente CliffWalking-v1 do Gymnasium.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import sys
from pathlib import Path

# Adiciona o diretório atual ao path para importações
sys.path.insert(0, str(Path(__file__).parent))

from sarsa_7b import sarsa
from expected_sarsa import expected_sarsa
from n_step_sarsa_9b import n_step_sarsa
from q_learning_8b import q_learning
from utils_rl import plotar_metricas, visualizar_politica, simular_trajetoria_gym, plot_trajetoria_gym


def treinar_todos(
    ambiente: str = 'CliffWalking-v1',
    episodios: int = 5000,
    alpha: float = 0.01,
    gamma: float = 0.9,
    epsilon: float = 0.1,
    n_steps: int = 3,
    seed: int = 42,
    salvar_resultados: bool = True
) -> Dict[str, Any]:
    """
    Treina todos os algoritmos de RL com os mesmos hiperparâmetros.
    
    Parâmetros
    ----------
    ambiente : str
        Nome do ambiente Gymnasium.
    episodios : int
        Número de episódios de treinamento.
    alpha : float
        Taxa de aprendizado.
    gamma : float
        Fator de desconto.
    epsilon : float
        Parâmetro ε da política ε-gulosa.
    n_steps : int
        Número de passos para N-step SARSA.
    seed : int
        Semente para reprodutibilidade.
    salvar_resultados : bool
        Se True, salva gráficos e visualizações.
    
    Retorna
    -------
    resultados : dict
        Dicionário com resultados de cada algoritmo.
    """
    # Cria ambiente
    env = gym.make(ambiente, render_mode='rgb_array')
    
    resultados = {}
    
    print("=" * 80)
    print("TREINAMENTO DE ALGORITMOS DE REINFORCEMENT LEARNING")
    print("=" * 80)
    print(f"Ambiente: {ambiente}")
    print(f"Episódios: {episodios}")
    print(f"Alpha: {alpha}, Gamma: {gamma}, Epsilon: {epsilon}")
    print("=" * 80)
    
    # 1. SARSA
    print("\n[1/4] Treinando SARSA...")
    env_sarsa = gym.make(ambiente, render_mode='rgb_array')
    Q_sarsa, Pi_sarsa, visitas_sarsa, _, T_sarsa, G_sarsa = sarsa(
        env_sarsa,
        gamma=gamma,
        N=episodios,
        epsilon=epsilon,
        alpha=alpha,
        seed=seed
    )
    env_sarsa.close()
    resultados['sarsa'] = {
        'Q': Q_sarsa,
        'Pi': Pi_sarsa,
        'visitas': visitas_sarsa,
        'T': T_sarsa,
        'G': G_sarsa
    }
    print(f"SARSA concluído. Retorno médio final: {np.mean(G_sarsa[-100:]):.2f}")
    
    # 2. Expected SARSA
    print("\n[2/4] Treinando Expected SARSA...")
    env_expected = gym.make(ambiente, render_mode='rgb_array')
    Q_expected, Pi_expected, visitas_expected, _, T_expected, G_expected = expected_sarsa(
        env_expected,
        gamma=gamma,
        N=episodios,
        epsilon=epsilon,
        alpha=alpha,
        seed=seed
    )
    env_expected.close()
    resultados['expected_sarsa'] = {
        'Q': Q_expected,
        'Pi': Pi_expected,
        'visitas': visitas_expected,
        'T': T_expected,
        'G': G_expected
    }
    print(f"Expected SARSA concluído. Retorno médio final: {np.mean(G_expected[-100:]):.2f}")
    
    # 3. N-step SARSA
    print("\n[3/4] Treinando N-step SARSA...")
    env_nstep = gym.make(ambiente, render_mode='rgb_array')
    Q_nstep, Pi_nstep, visitas_nstep, _, T_nstep, G_nstep = n_step_sarsa(
        env_nstep,
        n=n_steps,
        gamma=gamma,
        N=episodios,
        epsilon=epsilon,
        alpha=alpha,
        seed=seed
    )
    env_nstep.close()
    resultados['n_step_sarsa'] = {
        'Q': Q_nstep,
        'Pi': Pi_nstep,
        'visitas': visitas_nstep,
        'T': T_nstep,
        'G': G_nstep
    }
    print(f"N-step SARSA concluído. Retorno médio final: {np.mean(G_nstep[-100:]):.2f}")
    
    # 4. Q-learning
    print("\n[4/4] Treinando Q-learning...")
    env_qlearn = gym.make(ambiente, render_mode='rgb_array')
    Q_qlearn, Pi_qlearn, visitas_qlearn, _, T_qlearn, G_qlearn = q_learning(
        env_qlearn,
        gamma=gamma,
        N=episodios,
        epsilon=epsilon,
        alpha=alpha,
        seed=seed
    )
    env_qlearn.close()
    resultados['q_learning'] = {
        'Q': Q_qlearn,
        'Pi': Pi_qlearn,
        'visitas': visitas_qlearn,
        'T': T_qlearn,
        'G': G_qlearn
    }
    print(f"Q-learning concluído. Retorno médio final: {np.mean(G_qlearn[-100:]):.2f}")
    
    # Visualizações comparativas
    if salvar_resultados:
        print("\n" + "=" * 80)
        print("GERANDO VISUALIZAÇÕES...")
        print("=" * 80)
        
        # Gráfico comparativo de métricas
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Duração dos episódios
        janela = 100
        df = pd.DataFrame({
            'episodio': np.arange(episodios),
            'SARSA': T_sarsa,
            'Expected SARSA': T_expected,
            'N-step SARSA': T_nstep,
            'Q-learning': T_qlearn
        })
        
        for col in ['SARSA', 'Expected SARSA', 'N-step SARSA', 'Q-learning']:
            df[f'{col}_ma'] = df[col].rolling(window=janela).mean()
            axes[0].plot(df['episodio'], df[f'{col}_ma'], label=col, linewidth=2)
        
        axes[0].set_xlabel('Episódio')
        axes[0].set_ylabel('Passos por Episódio (média móvel)')
        axes[0].set_title('Comparação: Duração dos Episódios')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Retorno dos episódios
        df_ret = pd.DataFrame({
            'episodio': np.arange(episodios),
            'SARSA': G_sarsa,
            'Expected SARSA': G_expected,
            'N-step SARSA': G_nstep,
            'Q-learning': G_qlearn
        })
        
        for col in ['SARSA', 'Expected SARSA', 'N-step SARSA', 'Q-learning']:
            df_ret[f'{col}_ma'] = df_ret[col].rolling(window=janela).mean()
            axes[1].plot(df_ret['episodio'], df_ret[f'{col}_ma'], label=col, linewidth=2)
        
        axes[1].set_xlabel('Episódio')
        axes[1].set_ylabel('Retorno (média móvel)')
        axes[1].set_title('Comparação: Retorno dos Episódios')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('comparacao_algoritmos.png', dpi=150, bbox_inches='tight')
        print("Gráfico comparativo salvo: comparacao_algoritmos.png")
        plt.close()
        
        # Visualizações individuais
        algoritmos = {
            'SARSA': (Pi_sarsa, T_sarsa, G_sarsa),
            'Expected SARSA': (Pi_expected, T_expected, G_expected),
            'N-step SARSA': (Pi_nstep, T_nstep, G_nstep),
            'Q-learning': (Pi_qlearn, T_qlearn, G_qlearn)
        }
        
        for nome, (Pi, T, G) in algoritmos.items():
            print(f"\nVisualizando {nome}...")
            
            # Métricas
            plotar_metricas(T, G)
            plt.savefig(f'metricas_{nome.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # Política
            visualizar_politica(Pi, ambiente)
            plt.savefig(f'politica_{nome.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # Trajetória
            estados, _, _ = simular_trajetoria_gym(Pi, ambiente, max_steps=200)
            plot_trajetoria_gym(ambiente, estados, titulo=f"Trajetória - {nome}")
            plt.savefig(f'trajetoria_{nome.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    print("\n" + "=" * 80)
    print("TREINAMENTO CONCLUÍDO!")
    print("=" * 80)
    
    # Resumo final
    print("\nRESUMO DOS RESULTADOS:")
    print("-" * 80)
    for nome, dados in resultados.items():
        retorno_medio = np.mean(dados['G'][-100:])
        duracao_media = np.mean(dados['T'][-100:])
        print(f"{nome.upper():20s} | Retorno médio (últimos 100): {retorno_medio:7.2f} | Duração média: {duracao_media:6.2f}")
    print("-" * 80)
    
    env.close()
    return resultados


if __name__ == "__main__":
    # Hiperparâmetros
    EPISODIOS = 5000
    ALPHA = 0.01
    GAMMA = 0.9
    EPSILON = 0.1
    N_STEPS = 3
    SEED = 42
    
    # Treina todos os algoritmos
    resultados = treinar_todos(
        ambiente='CliffWalking-v1',
        episodios=EPISODIOS,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,
        n_steps=N_STEPS,
        seed=SEED,
        salvar_resultados=True
    )
