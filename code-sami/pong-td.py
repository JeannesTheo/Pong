import cv2
import gymnasium as gym
import numpy as np
import shimmy
import tensorflow as tf
from ale_py import ALEInterface
from ale_py.roms import Pong
from gymnasium.utils.play import play
from keras.src.saving.saving_api import load_model
from matplotlib.font_manager import os
from matplotlib.pyplot import Enum, np
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers

ale = ALEInterface()

ale.loadROM(Pong)
env = gym.make("ALE/Pong-v5", render_mode="human")
# env = gym.make("ALE/Pong-v5")
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.

num_actions = 3

# model = load_model("./models/actor-critic-90.h5")
model = load_model("./reinforcement-learning/models/pong-21.h5")


class PongActions(Enum):
    NO_ACTION = 0
    UP = 2
    DOWN = 3

    def to_categorical_index(self):
        if self == PongActions.NO_ACTION:
            return 0
        if self == PongActions.UP:
            return 1
        if self == PongActions.DOWN:
            return 2
        return -1

    @staticmethod
    def from_categorical_index(num):
        if num == 0:
            return PongActions.NO_ACTION.value
        if num == 1:
            return PongActions.UP.value
        if num == 2:
            return PongActions.DOWN.value


OBS_PATH = "data/obs.csv"
X_PATH = "data/X.csv"
X_COORD_PATH = "data/X_coord.csv"
Y_PATH = "data/y.csv"
L_PLAYER_POS = 0
R_PLAYER_POS = 61
UNK_POSITION = np.array([255, 255])


class Observation:
    def __init__(self, obs_t, obs_tp1) -> None:
        self.obs = Observation.__crop__(obs_t)
        # print(self.obs.shape)
        obs_t_preprocessed = Observation.__preprocess__(obs_t)
        left_player = (
            obs_t_preprocessed[0]
            if obs_t_preprocessed[0][1] == L_PLAYER_POS
            else UNK_POSITION
        )

        right_player_index = (np.where(obs_t_preprocessed[:, 1] == R_PLAYER_POS))[0][0]

        right_player = obs_t_preprocessed[right_player_index]

        ball_t = obs_t_preprocessed[right_player_index - 1]
        self.ball_t = (
            ball_t
            if ball_t[1] != L_PLAYER_POS and ball_t[1] != R_PLAYER_POS
            else UNK_POSITION
        )

        obs_tp1_preprocessed = Observation.__preprocess__(obs_tp1)
        right_player_index = (np.where(obs_tp1_preprocessed[:, 1] == R_PLAYER_POS))[0][
            0
        ]
        ball_tp1 = obs_tp1_preprocessed[right_player_index - 1]
        self.ball_tp1 = (
            ball_tp1
            if ball_tp1[1] != L_PLAYER_POS and ball_tp1[1] != R_PLAYER_POS
            else UNK_POSITION
        )

        self.data = np.array([left_player, self.ball_t, self.ball_tp1, right_player])

    def add_action(self, action):
        self.action = [PongActions(action).to_categorical_index()]

    def is_ball_going_towards_enemy(self):
        return self.ball_t[1] > self.ball_tp1[1]

    def save(self):
        with open(OBS_PATH, "a") as outfile_X:
            np.savetxt(outfile_X, delimiter=",", X=[self.obs.flatten()], fmt="%d")
        with open(X_PATH, "a") as outfile_X:
            np.savetxt(outfile_X, delimiter=",", X=[self.data.flatten()], fmt="%d")
        with open(Y_PATH, "a") as outfile_Y:
            np.savetxt(outfile_Y, delimiter=",", X=self.action, fmt="%d")

    @staticmethod
    def __preprocess__(obs):
        cropped = Observation.__crop__(obs)
        elements = np.argwhere(cropped == 1)
        return np.array(sorted(elements, key=lambda x: x[1]))

    @staticmethod
    def __crop__(obs):
        return (obs[34:194:4, 40:142:2, 2] > 50).astype(np.uint8).astype(float)


print("loaded")
# print(model.summary())
state_before = env.reset()[0]
state_before = Observation.__crop__(state_before).reshape(-1, 40, 51, 1)
state = None
env.render()
while True:
    if state is None or state_before is None:
        action = env.action_space.sample()
        state = state_before
        (state_before, reward, terminated, *rest) = env.step(action)
        state_before = (Observation.__crop__(state_before)).reshape(-1, 40, 51, 1)
        continue

    # print("state shap", state.shape)
    # print("state_before shap", state_before.shape)

    # state_before_copy = state_before.copy()
    # state_before_copy[0][:, -1] = 0
    state[0][:, -1] = 0
    state = state_before - state
    # im = Image.fromarray(state[0].reshape(40, 51))
    # im.save("./test.png")
    # state = tf.convert_to_tensor(state)
    # state = tf.expand_dims(state, 0)

    output_dir = "output_images_test"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, img_matrix in enumerate(state):
        # Create an empty 3-channel image of the same dimensions
        rgb_image = np.zeros(
            (img_matrix.shape[0], img_matrix.shape[1], 3), dtype=np.uint8
        )

        # Set the red channel to 255 wherever img_matrix is 1
        rgb_image[img_matrix[:, :, 0] == -1, 0] = 255

        # Set the blue channel to 255 wherever img_matrix is -1
        rgb_image[img_matrix[:, :, 0] == 1, 2] = 255

        filename = os.path.join(output_dir, f"image_{idx}.png")
        cv2.imwrite(filename, rgb_image)

    # if state[0][:, -1, 0].any() != 0:
    #     print("visible")
    #     print(state[0][:, -1, 0])
    # else:
    #     print(state[0][:, -1, 0])

    action = model.predict(state, verbose=False)
    # action_probs, critic_value = model.predict(state)
    # print(action)
    action = np.argmax(action)
    # print(action)
    # action = np.random.choice(num_actions, p=np.squeeze(action_probs))
    # print(state.shape)
    # if state[0][39][50][0] != 0:
    #     print("in corner")
    action = PongActions.from_categorical_index(action)
    print(action)

    state = state_before
    (state_before, reward, terminated, *rest) = env.step(action)
    state_before = (Observation.__crop__(state_before)).reshape(-1, 40, 51, 1)
    # print(reward)
