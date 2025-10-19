# %%
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# %%
# Create environment
env = gym.make("CliffWalking-v1", render_mode="rgb_array")
obs, info = env.reset()

# %%
# Show initial state
img = env.render()
plt.imshow(img)
plt.axis('off')
plt.title('Initial State')
plt.show()

# %%
# Manually win the game - simple path
print("Winning the game...")
obs, info = env.reset()  # Reset to start

# Simple winning path: 3 up, then 9 right, then 1 up
actions = [0, 0, 0] + [2] * 9 + [0]  # up, up, up, right x9, up

for i, action in enumerate(actions):
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Show each step
    img = env.render()
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Step {i+1}: Action {action}, Reward: {reward}')
    plt.show()
    
    print(f"Step {i+1}: Action {action}, State: {obs}, Reward: {reward}")
    
    if terminated or truncated:
        print(f"Episode finished! Total reward: {reward}")
        break

env.close()
# %%
