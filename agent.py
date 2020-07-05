import torch
import gym
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any
from random import sample, random
import wandb
from tqdm import tqdm
from collections import deque
import numpy as np

@dataclass
class Sarsd:
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool

class DQNAgent:
    def __init__(self, model):
        self.model=  model

    def get_actions(self, observations):
        #obs shape: (N batches, 4)
        q_vals = self.model(observations)

        #q vals shape: (N, 2 - action number)
        return q_vals.max(-1)[1]

#model sieci
class Model(nn.Module):
    def __init__(self, obs_shape, num_actions):
        super(Model, self).__init__()
        assert len(obs_shape) == 1, "This network works only for flat observations"
        self.obs_shape = obs_shape
        self.num_action = num_actions
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_shape[0], 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_actions),
        )
        self.opt = optim.Adam(self.net.parameters(), lr = 0.0001)

    def forward(self, x):
        return self.net(x)

#pamiec gry, wykonane akcje itd /// state>action>reward>next state
class ReplayBuffer:
    def __init__(self, buffer_size=100000):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen = buffer_size)

    def insert(self, sarsd):
        self.buffer.append(sarsd)
        #self.buffer = self.buffer[-self.buffer_size:]

    def sample(self, num_samples):
        assert num_samples <= len(self.buffer)
        return sample(self.buffer, num_samples)

#kopiowanie wag polaczen z jednej sieci do drugiej
def update_tgt_model(m, tgt):
    tgt.load_state_dict(m.state_dict())

def train_step(model, state_transitions, tgt, num_actions):
    cur_states = torch.stack(([torch.Tensor(s.state) for s in state_transitions]))
    rewards = torch.stack(([torch.Tensor([s.reward]) for s in state_transitions]))
    mask = torch.stack(([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions]))
    next_states = torch.stack(([torch.Tensor(s.next_state) for s in state_transitions]))
    actions = [s.action for s in state_transitions]

    with torch.no_grad(): # nie zapisuje do sieci tgt
        qvals_next = tgt(next_states).max(-1)[0] # shape -> (N, num_actions) /obliczenie q vals dla next action

    model.opt.zero_grad()
    qvals = model(cur_states) # oblicza q val
    one_hot_actions = F.one_hot(torch.LongTensor(actions), num_actions)

    loss = ((rewards + mask[:, 0]*qvals_next - torch.sum(qvals*one_hot_actions, -1))**2).mean()
    loss.backward()
    model.opt.step()
    return loss

def main(test = False, chkpt = None):
    if not test:
        wandb.init(project='dqn-tutorial', name='dqn-cartpole')
    min_rb_size = 10000
    sample_size = 2500

    eps_min = 0.01
    eps_decay = 0.999995

    env_steps_before_train = 100
    tgt_model_update = 150

    env = gym.make("CartPole-v1")
    last_observation = env.reset()

    m = Model(env.observation_space.shape, env.action_space.n)
    if chkpt is not None:
        m.load_state_dict(torch.load(chkpt))
    tgt = Model(env.observation_space.shape, env.action_space.n)
    update_tgt_model(m, tgt)

    rb = ReplayBuffer()
    steps_since_train = 0
    epochs_since_tgt = 0

    step_num = -1 * min_rb_size

    episode_rewards = []
    rolling_reward = 0

    tq = tqdm()
    try:
        while True:
            if test:
                env.render()
                time.sleep(0.01)
            tq.update(1)
            eps = eps_decay**(step_num)
            if test:
                eps = 0

            if random() < eps:
                action = env.action_space.sample()
            else:
                action = m(torch.Tensor(last_observation)).max(-1)[-1].item()

            observation, reward, done, info = env.step(action)
            rolling_reward += 1
            reward = reward/100.0

            rb.insert(Sarsd(last_observation, action, reward, observation, done))
            last_observation = observation

            if done:
                episode_rewards.append(rolling_reward)
                if test:
                    print(rolling_reward)
                rolling_reward = 0
                observation = env.reset()

            steps_since_train += 1
            step_num += 1

            if (not test) and len(rb.buffer) > min_rb_size and steps_since_train > env_steps_before_train:
                loss = train_step(m, rb.sample(sample_size), tgt, env.action_space.n)
                wandb.log({'loss': loss.detach().item(), 'eps': eps,'avg_reward': np.mean(episode_rewards)}) # wysylanie logow do wandb app
                episode_rewards = []
                #print(step_num, loss.detach().item())
                epochs_since_tgt += 1
                if epochs_since_tgt > tgt_model_update:
                    print('updating tgt model')
                    update_tgt_model(m, tgt)
                    epochs_since_tgt = 0
                    torch.save(tgt.state_dict(), f"/home/karol/Dokumenty/Magisterka badania/cartpole_test/trained_models/{step_num}.pth")
                steps_since_train = 0
    except KeyboardInterrupt:
        pass

    env.close()

if __name__ == '__main__':
    #main(True, "/home/karol/Dokumenty/Magisterka badania/cartpole_test/trained_models/trained.pth")
    main()
    #test commit mode