import collections
import random
from functools import reduce

import numpy as np
import tensorflow as tf

from tensorflow.keras import Model, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

from .parameters import device # TODO

Experience = collections.namedtuple("Experience", ("state", "action", "reward", "next_state", "done"))


class ReplayMemory:
  def __init__(self, capacity, batch_size):
    self.states = collections.deque([], maxlen=capacity)
    self.actions = collections.deque([], maxlen=capacity)
    self.rewards = collections.deque([], maxlen=capacity)
    self.next_states = collections.deque([], maxlen=capacity)
    self.dones = collections.deque([], maxlen=capacity)
    self.size = 0
    self.capacity = capacity
    self.batch_size = batch_size  

  def __len__(self):
    #return len(self.memory)
    return self.size

  def push(self, state, action, reward, next_state, done):
    #if len(self.memory) < self.capacity:
    #  self.memory.append(None)
    #self.memory[self.position] = (state, action, reward, next_state, done)
    #self.position = (self.position + 1) % self.capacity
    self.states.append(state)
    self.actions.append(action)
    self.rewards.append(reward)
    self.next_states.append(next_state)
    self.dones.append(done)
    self.size = min(self.size + 1, self.capacity)

  def sample(self):
    assert self.size >= self.batch_size
    indices = random.sample(range(self.size), k=self.batch_size)
    exps = (self.states, self.actions, self.rewards, self.next_states)
    extract = lambda list_: [list_[i] for i in indices]
    functions = (extract, np.vstack, torch.from_numpy) # TODO
    states, actions, rewards, next_states = reduce(lambda x, y: map(y, x), functions, exps)
    rewards = rewards.reshape(-1)
    dones = tf.cast(np.vstack(self.dones), tf.int32) # Note: We return all the data types as tf.float32 so that they all get converted by TF
    tofloat = lambda x: x.float().to(device)
    tolong = lambda x: x.long().to(device)
    return (
        tofloat(states),
        tolong(actions),
        tofloat(rewards),
        tofloat(next_states),
        tf.cast(dones, tf.float32),
    )


    
