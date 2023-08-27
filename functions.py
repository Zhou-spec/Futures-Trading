import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# write a function to get a path from the policy network
def get_path(policy_network, train_dataset):
    holding_positions = [0]
    pos_changes = []
    rewards = []
    states1 = []
    states2 = [] 
    action_dist_set = []
    action_set = []
    for i in range(len(train_dataset)):
        bid_ask = train_dataset[i:i + 1, :]
        bid_ask = torch.tensor(bid_ask).float().to(device)
        bid_ask = bid_ask.squeeze(0)
        old_hold = holding_positions[-1]
        hold = torch.tensor([old_hold]).float().to(device)
        states1.append(bid_ask)
        states2.append(hold)
        # get the action from the policy network, which is a probability distribution
        
        action_dist = policy_network(bid_ask, hold)
        action_dist_set.append(action_dist)
        # sample an action from the probability distribution
        action = torch.multinomial(action_dist, 1).item() - 1
        action_set.append(action + 1)
        
        # decide the position change based on the action and current holding position
        if old_hold == 0:
            # make the action to be integer
            a = int(action)
            holding_positions.append(a)
        if old_hold == 1:
            holding_positions.append(min(action + old_hold, old_hold))
        if old_hold == -1:
            holding_positions.append(max(action + old_hold, old_hold))
           
        # compute the reward (cash change) 
        new_hold = holding_positions[-1]
        position_change = new_hold - old_hold
        pos_changes.append(position_change)
    
        reward = 0
        if position_change == 0:
            reward = 0
        if position_change == 1:
            reward = - train_dataset[i][2] 
        if position_change == -1:
            reward = train_dataset[i][0] 

        rewards.append(reward)
    
    # at the end, the agent needs to liquid all positions
    if holding_positions[-1] == 1:
        rewards.append(train_dataset[-1][2])
    elif holding_positions[-1] == -1:
        rewards.append(-train_dataset[-1][0])
    else:
        rewards.append(0)

    # convert the list of rewards to a tensor
    rewards = torch.tensor(rewards).float().to(device)
    # convert the list of states to a tensor
    states1 = torch.stack(states1).to(device)
    states2 = torch.stack(states2).to(device)
    action_dist_set = torch.stack(action_dist_set).to(device)
    return states1, states2, rewards, action_dist_set, action_set


        
# train the value network using the TD(0) algorithm
def train_value_network(value_network, policy_network, train_dataset, value_optimizer, gamma = 0.99, epochs = 20):
    # define the loss function
    loss = 0
    losses = []
    # loop over the epochs
    for epoch in range(epochs):
        # get the path from the policy network
        # only use one trajectory each epoch
        states1, states2, rewards, action_dist_set, action_set = get_path(policy_network, train_dataset)
        # get the value estimate from the value network
        value_estimate = []
        for i in range(len(states1)):
            value_estimate.append(value_network(states1[i], states2[i]))
        value_estimate = torch.stack(value_estimate).squeeze(1)
        new_value_estimate = torch.cat((value_estimate, torch.tensor([0]).to(device)))
        # compute the TD(0) error
        loss = (rewards[:-1] + gamma * new_value_estimate[1:] - new_value_estimate[:-1]).pow(2).mean()
        # zero the gradient
        value_optimizer.zero_grad()
        # compute the gradient
        loss.backward()
        # update the weights
        value_optimizer.step()
        # print the loss
        losses.append(loss.item())
        if (epoch - 4) % 5 == 0:
            print('value epoch: {}, value loss: {:.5f}'.format(epoch + 1, loss.item()))

    #plt.figure(figsize = (10, 5))
    #plt.plot(losses)
    #plt.xlabel('Epoch')
    #plt.ylabel('Loss')
    #plt.title('Loss over Epochs')
    #plt.show()
    return losses


# the following define a function that compute advantage estimation for a trajectory
def advantage_estimate(states1, states2, rewards, value_network, gamma):
    value_estimate = []
    for i in range(len(states1)):
        value_estimate.append(value_network(states1[i], states2[i]))
    value_estimate = torch.stack(value_estimate).squeeze(1)
    new_value_estimate = torch.cat((value_estimate, torch.tensor([0]).to(device)))
    advantages = rewards[:-1] + gamma * new_value_estimate[1:] - new_value_estimate[:-1]
    return advantages


        

# states, rewards, actions are trajetory data of old policy
# there is a new_policy_network that is updated by ppo_update()
def ppo_loss(new_policy_network, policy_network, value_network, train_dataset, batch_size, epsilon=0.2, gamma = 0.99):
    # batch size: the number of trajectories
    loss = torch.tensor(0.0, requires_grad=True).to(device)
    for _ in range(batch_size):
        states1, states2, rewards, action_dist_set, action_set = get_path(policy_network, train_dataset)
        new_action_dist_set = [new_policy_network(states1[i], states2[i]) for i in range(len(states1))]
        new_action_dist_set = torch.stack(new_action_dist_set).to(device)
        action_dist_set = action_dist_set.detach()
        rewards = rewards.detach()
        ratio = []
        for i in range(len(action_dist_set)):
            ratio.append(new_action_dist_set[i][action_set[i]] / action_dist_set[i][action_set[i]])
        ratio = torch.stack(ratio).to(device).detach()
        # compute the advantage of the trajectory
        advantage = advantage_estimate(states1, states2, rewards, value_network, gamma)
        advantage = advantage.detach()
        # compute the clipped ratio
        clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
        # compute the surrogate loss
        policy_loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()
        # compute the total loss
        loss = loss + policy_loss
    return loss

    
def ppo_train(new_policy_network, policy_network, value_network, optimizer, train_dataset, batch_size, epochs, epsilon=0.1, gamma = 0.99):
    
    for epoch in range(epochs):
        loss = ppo_loss(new_policy_network, policy_network, value_network, train_dataset, batch_size, epsilon, gamma)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return None
    


def wealth_dist(num_traj, policy_network, dataset):
    wealths = []
    for num in range(num_traj):
        states1, states2, rewards, action_dist_set, action_set = get_path(policy_network, dataset)
        wealths.append(sum(rewards))

    return wealths

