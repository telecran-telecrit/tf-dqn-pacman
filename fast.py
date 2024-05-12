import pickle, random
import matplotlib.pyplot as plt # from matplotlib import pyplot as plt
from pathlib import Path
from tensorflow.keras.models import load_model

import tensorflow as tf
from utils import ACTIONS
import cv2
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path = str(Path().absolute())

def moving_average(values, n):
    offset = (n - 1) // 2
    v = [values[0]] * offset + values + [values[-1]] * offset
    return [
        sum(v[i - offset : i + offset + 1]) / n for i in range(offset, len(v) - offset)
    ]


def only_rewards(filename):
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
    # Set environment
    ale = ALEInterface()
    ale.loadROM(Pacman)
    env = gym.make("MsPacman-v0")

    agent = DQN(N_ACTIONS)
    agent.load_weights(str(filename))

    dmaker = DecisionMaker(0, agent)
    obs = env.reset()

    frame_size = (160, 210)
    path_video = path / "output_video.avi"
    bin_loader = cv2.VideoWriter_fourcc(*"DIVX")
    out = cv2.VideoWriter(str(path_video), bin_loader, 30, frame_size)

    # Avoid beginning steps of the game
    for i_step in range(AVOIDED_STEPS):
        obs, reward, done, info = env.step(3)

    observations = init_obs(env)
    obs, reward, done, info = env.step(3)
    out.write(cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
    
    old_action = 3
    pass
    while True:
        state = preprocess_observation(observations, obs)
        
        action = DQN(np.expand_dims(state, axis=0)).numpy().argmax() # action = np.argmax(agent(state.astype(np.float32)))
        
        action_ = ACTIONS[old_action][action.item()]
        obs, reward, done, info = env.step(action)
        out.write(cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
        old_action = action_
        if done:
            break

    out.release()
    print('"output_video.avi" saved in {}'.format(path))
    
# only_rewards("episode-700.pkl")
record(path / "policy-model-700.pt")
