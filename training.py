import gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from policy import Policy
import time

N_EPISODES = 1000
MAX_TIMESTEPS = 1000
GAMMA = 1.0
PRINT_EVERY = 100
GOAL = 195.0
LEARNING_RATE = 0.005
DEVICE = 'cpu'

def get_policy_loss(log_probabilities, traj_rewards_sum):
    policy_loss = []
    
    for log_prob in log_probabilities:
        policy_loss.append(-log_prob * traj_rewards_sum)
    
    # Concatenates the given sequence of seq tensors in the given dimension.
    return torch.cat(policy_loss).sum()

def get_trajectory_rewards(trajectory_rewards):
    discounts = [GAMMA**i for i in range(len(trajectory_rewards)+1)]
    return sum([a*b for a,b in zip(discounts, trajectory_rewards)])

def back_and_step_forward(optimizer, policy_loss):
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

def do_episode(env, policy):
    saved_log_probs = []
    trajectory_rewards = []
    state = env.reset()
    for t in range(MAX_TIMESTEPS):
        action, log_probabilities = policy.act(state, DEVICE)
        state, reward, done, _ = env.step(action)
        trajectory_rewards.append(reward)
        saved_log_probs.append(log_probabilities)     
        if done:
            break
         
    traj_rewards_sum = get_trajectory_rewards(trajectory_rewards)
    policy_loss = get_policy_loss(saved_log_probs, traj_rewards_sum)
    back_and_step_forward(optimizer, policy_loss)
    
    return trajectory_rewards

def training(env, policy, optimizer):
    scores_deque = deque(maxlen=PRINT_EVERY)
    scores = []
    
    for i_episode in range(1, N_EPISODES+1):
        trajectory_rewards = do_episode(env, policy)
        scores_deque.append(sum(trajectory_rewards))
        scores.append(sum(trajectory_rewards))
    
        if i_episode % PRINT_EVERY == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque)>= GOAL:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                i_episode-PRINT_EVERY, np.mean(scores_deque)))
            timestr = time.strftime("%Y%m%d-%H%M%S")
            torch.save(policy.state_dict(), 'trained_policy_{}.pth'.format(timestr))
            break
    
    return scores

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    # print('Observation Space:', env.observation_space)
    # print('Action Space:', env.action_space)
    
    policy = Policy().to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    scores = training(env, policy, optimizer)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()