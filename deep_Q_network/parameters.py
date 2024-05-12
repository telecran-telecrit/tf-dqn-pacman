import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from typing import Tuple, Callable, Union

# Preprocessing constant
WALL_COLOR = [228, 111, 111]
BACKGROUND = [0, 28, 136]
PACMAN_COLOR = [210, 164, 74]

def tonumpy(color: Tuple):
  """Convert color to numpy array.

  Args:
    color: The color value.

  Returns:
    A numpy array with the shape (1, 1, 3).
  """
  return np.array([[color]], dtype=np.uint8)


def togray(color: Union[Tuple, str]) -> int:
  """Convert color to grayscale.

  Args:
    color: The color value or image path.

  Returns:
    The grayscale value.
  """
  if isinstance(color, str):
    color = np.array(load_img(color, grayscale=True))
  else:
    color = tonumpy(color)
    color = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)

  return color[0][0]


WALL_COLOR_GRAY = togray(WALL_COLOR)
BACKGROUND_GRAY = togray(BACKGROUND)
PACMAN_COLOR_GRAY = togray(PACMAN_COLOR)

# Reinforcement learning constants
BATCH_SIZE = 128
DISCOUNT_RATE = 0.99
EPS_MAX = 1.0
EPS_MIN = 0.1
EPS_DECAY = 1_000_000
TARGET_UPDATE = 8_000
REPLAY_MEMORY_SIZE = 2 * 6_000

# Environment constants
N_ACTIONS = 4
AVOIDED_STEPS = 80  # at the beginning, there is a period of time where the game doesn't allowed the player to move Pacman
DEAD_STEPS = 36  # frames to avoid when the agent dies
K_FRAME = 2

# Optimizer parameters
LEARNING_RATE = 2.5e-4
# DECAY_RATE = 0.99
MOMENTUM = 0.95

# Algorithm constant
MAX_FRAMES = 2_000_000
SAVE_MODEL = 20  # episodes
