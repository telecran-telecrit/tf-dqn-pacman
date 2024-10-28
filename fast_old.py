import pickle, torch, random
from pathlib import Path
from deep_Q_network_old import *
from utils_old import ACTIONS
import gym

path = Path().absolute()

import random, math, torch
from .parameters import EPS_MAX, EPS_MIN, EPS_DECAY, N_ACTIONS, device
from utils_old.utils import ACTIONS, REVERSED

class DecisionMaker: # in new version it was replaced by the method DataHandler.select_action
    def __init__(self, steps_done, policy_DQN):
        self.steps_done = steps_done
        self.old_action = 3

    def select_action(self, state, policy_DQN, display, learn_counter):
        sample = random.random()
        eps_threshold = max(EPS_MIN, EPS_MAX - (EPS_MAX - EPS_MIN) * learn_counter / EPS_DECAY)
        self.steps_done += 1
        with torch.no_grad():
            q_values = policy_DQN(state)
        display.data.q_values.append(q_values.max(1)[0].item()) # old way to collect data for visualization
        if sample > eps_threshold:
            # Optimal action
            return q_values.max(1)[1].view(1, 1)
        else:
            # Random action
            action = random.randrange(N_ACTIONS)
            while action == REVERSED[self.old_action]:
                action = random.randrange(N_ACTIONS)
            return torch.tensor([[action]], device=device, dtype=torch.long)

def moving_average(values, n):
    offset = (n - 1) // 2
    v = [values[0]] * offset + values + [values[-1]] * offset
    return [
        sum(v[i - offset : i + offset + 1]) / n for i in range(offset, len(v) - offset)
    ]


def only_rewards(filename):
    from matplotlib import pyplot as plt

    Y_LABELS = (
        "Average of rewards per episode",
        "Total of rewards per episode",
    )

    with open(path / filename, "rb") as file:
        data = pickle.load(file)

    iterations = list(map(range, map(len, data[:-1])))

    fig, axis = plt.subplots(2, 1, figsize=(16, 10))
    successes = data[-1]

    for i, it in enumerate((1, 4)):
        axis[i].plot(iterations[it][4:], data[it][4:])
        axis[i].plot(iterations[it][4:], moving_average(data[it][4:], 17))
    for label, axis in zip(Y_LABELS, axis):
        axis.set_ylabel(label)
    fig.tight_layout()
    plt.savefig("rewards.png")


def record(filename):
    import cv2

    # Set environment
    ##ale = ALEInterface()
    ##ale.loadROM(Pacman)
    env = gym.make("MsPacman-v0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = DQN(N_ACTIONS).to(device)

    agent.load_state_dict(torch.load(str(filename), map_location=device))
    agent.eval()

    #dmaker = DecisionMaker(0, agent)
    obs = env.reset()

    frameSize = (160, 210)
    path_video = path / "output_video.avi"
    bin_loader = cv2.VideoWriter_fourcc(*"DIVX")
    out = cv2.VideoWriter(str(path_video), bin_loader, 30, frameSize)

    # Avoid beginning steps of the game
    for i_step in range(AVOIDED_STEPS):
        obs, reward, done, info = env.step(3)

    observations = init_obs(env)
    obs, reward, done, info = env.step(3)
    out.write(cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
    old_action = 3

    while True:
        state = preprocess_observation(observations, obs)
        # sample = random.random()
        # eps_threshold = EPS_MIN
        # with torch.no_grad():
        #     q_values = agent(state)
        # if sample > eps_threshold:
        #     action = q_values.max(1)[1].view(1, 1)
        # else:
        #     random_action = [[random.randrange(N_ACTIONS)]]
        #     action = torch.tensor(random_action, device=device, dtype=torch.long)
        action = agent(state).max(1)[1].view(1, 1)

        action_ = ACTIONS[old_action][action.item()]
        obs, reward, done, info = env.step(action)
        out.write(cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
        old_action = action_
        if (done and info["ale.lives"] == 0):
            break

    out.release()
    print('"output_video.avi" saved in {}'.format(path))


# only_rewards("episode-560.pkl")
record(path / "policy-model-560.pt")
