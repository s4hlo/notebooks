# %%
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# %%

def print_state(env):
    img = env.render()
    plt.imshow(img)
    plt.axis('off')
    plt.show()
# %%
# Create environment
env = gym.make("CliffWalking-v1", render_mode="rgb_array", is_slippery=False)
env = env.unwrapped

# %%
# up -> 9 * right -> down
# this execute a action
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(0)
for i in range(11):
    obs, reward, terminated, truncated, info = env.step(1)
# obs, reward, terminated, truncated, info = env.step(2)


print("state apos step: ", obs)

print("reward: ", reward)

print("terminou o episodio: ", terminated)

print("proxima ação: ", info)

print_state(env)
# %%


# %%
# SARSA Algorithm Implementation

def sarsa(env, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    """
    SARSA algorithm for Cliff Walking environment
    
    Args:
        env: Gymnasium environment
        episodes: Number of training episodes
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate
    """
    # Initialize Q-table
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    env.observation_space.n
    Q = np.zeros((n_states, n_actions))
    
    # Track rewards for plotting
    episode_rewards = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        state = obs
        total_reward = 0
        
        # Choose initial action using epsilon-greedy
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        
        while True:
            # Take action and observe next state and reward
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = next_obs
            total_reward += reward
            
            # Choose next action using epsilon-greedy
            if np.random.random() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state])
            
            # SARSA update
            Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
            
            state = next_state
            action = next_action
            
            if terminated or truncated:
                break
        
        episode_rewards.append(total_reward)
        
        # Print progress
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
    
    return Q, episode_rewards

# %%
# Train SARSA agent
print("Training SARSA agent...")
Q_table, rewards = sarsa(env, episodes=1000)

# %%
# Test the trained agent
def test_agent(env, Q_table, episodes=10):
    """Test the trained agent"""
    for episode in range(episodes):
        obs, info = env.reset()
        state = obs
        total_reward = 0
        steps = 0
        
        print(f"\nEpisode {episode + 1}:")
        print(f"Starting state: {state}")
        
        while steps < 100:  # Prevent infinite loops
            action = np.argmax(Q_table[state])
            obs, reward, terminated, truncated, info = env.step(action)
            state = obs
            total_reward += reward
            steps += 1
            
            print(f"Step {steps}: Action {action}, Reward {reward}, New state {state}")
            
            if terminated or truncated:
                break
        
        print(f"Total reward: {total_reward}, Steps: {steps}")

# %%
# Plot training progress
plt.figure(figsize=(10, 6))
plt.plot(rewards)
plt.title('SARSA Training Progress')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)
plt.show()

# %%
# Test the trained agent
test_agent(env, Q_table, episodes=3)

# %%
# Show final Q-table
print("\nFinal Q-table (first 10 states):")
print(Q_table[:10])

Q_table.shape

# %%
# Visualizar a política aprendida
def visualize_policy(env, Q_table):
    """
    Visualiza a política aprendida no ambiente Cliff Walking
    """
    # Obter as dimensões do grid
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    # Para CliffWalking-v1: 4x12 grid
    height = 4
    width = 12
    
    # Criar matriz para armazenar as ações
    policy_grid = np.zeros((height, width), dtype=int)
    
    # Preencher a política
    for state in range(n_states):
        row = state // width
        col = state % width
        action = np.argmax(Q_table[state])
        policy_grid[row, col] = action
    
    # Criar visualização
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Mapear ações para símbolos
    action_symbols = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    action_names = {0: 'UP', 1: 'RIGHT', 2: 'DOWN', 3: 'LEFT'}
    
    # Criar matriz de símbolos
    symbol_grid = np.empty((height, width), dtype=object)
    for i in range(height):
        for j in range(width):
            symbol_grid[i, j] = action_symbols[policy_grid[i, j]]
    
    # Plotar o grid
    im = ax.imshow(policy_grid, cmap='viridis', aspect='equal')
    
    # Adicionar símbolos das ações
    for i in range(height):
        for j in range(width):
            ax.text(j, i, symbol_grid[i, j], ha='center', va='center', 
                   fontsize=16, fontweight='bold', color='white')
    
    # Configurar eixos
    ax.set_xticks(range(width))
    ax.set_yticks(range(height))
    ax.set_xticklabels(range(width))
    ax.set_yticklabels(range(height))
    ax.set_xlabel('Coluna')
    ax.set_ylabel('Linha')
    ax.set_title('Política Aprendida pelo SARSA\n↑=UP, →=RIGHT, ↓=DOWN, ←=LEFT')
    
    # Adicionar barra de cores
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Ação')
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['UP', 'RIGHT', 'DOWN', 'LEFT'])
    
    plt.tight_layout()
    plt.show()
    
    return policy_grid

# %%
# Visualizar a política
print("Política aprendida:")
policy = visualize_policy(env, Q_table)

# %%
# Mostrar estatísticas da política
print(f"\nDistribuição das ações na política:")
unique, counts = np.unique(policy, return_counts=True)
action_names = {0: 'UP', 1: 'RIGHT', 2: 'DOWN', 3: 'LEFT'}
for action, count in zip(unique, counts):
    print(f"{action_names[action]}: {count} estados")

# %%
