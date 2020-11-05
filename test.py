import gym
import torch
from policy import Policy

DEVICE = 'cpu'
env = gym.make('CartPole-v0')
env._max_episode_steps = 10000
env = gym.wrappers.Monitor(env, "./monitor_output", force=True)

policy = Policy()
policy.load_state_dict(torch.load('trained_policy_20201105-135133.pth'))

state = env.reset()
for _ in range(10000):
    action, _ = policy.act(state, DEVICE)
    state, reward, done, _ = env.step(action)
    if done:
        break
env.close()