import gym
import torch
from policy import Policy

env = gym.make('CartPole-v0')
env = gym.wrappers.Monitor(env, "./monitor_output", force=True)

policy = Policy()
policy.load_state_dict(torch.load('trained_policy.pth'))

state = env.reset()
for _ in range(5000):
    action = policy.act(state)
    state, reward, done, _ = env.step(action)
    if done:
        break 
            
env.close()