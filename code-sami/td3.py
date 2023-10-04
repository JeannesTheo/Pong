from random import random

import gymnasium as gym
from ale_py import ALEInterface
from gymnasium.utils.play import play
from matplotlib.pyplot import Enum, np

ale = ALEInterface()

from ale_py.roms import Pong


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


OBS_PATH = "data/obs.csv"
DIFF_PATH = "data/diff.csv"
X_PATH = "data/X.csv"
X_COORD_PATH = "data/X_coord.csv"
Y_PATH = "data/y.csv"
L_PLAYER_POS = 0
R_PLAYER_POS = 50
UNK_POSITION = np.array([255, 255])


class Observation:
    def __init__(self, obs_t, obs_tp1) -> None:
        self.obs = Observation.__crop__(obs_t)
        self.obs_tp1 = Observation.__crop__(obs_tp1)
        # print(self.obs.shape)
        obs_t_preprocessed = Observation.__preprocess__(obs_t)
        # left_player = (
        #     obs_t_preprocessed[0]
        #     if obs_t_preprocessed[0][1] == L_PLAYER_POS
        #     else UNK_POSITION
        # )

        # print(obs_t_preprocessed)
        right_player_index = (np.where(obs_t_preprocessed[:, 1] == R_PLAYER_POS))[0][0]

        right_player = obs_t_preprocessed[right_player_index]
        # print(obs_t_preprocessed[right_player_index])

        ball_t = obs_t_preprocessed[right_player_index - 1]
        # print(right_player_index)
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

        self.data = np.array([self.ball_t, self.ball_tp1, right_player])

    def add_action(self, action):
        # if action == 0:
        # print("action 0")
        self.action = [PongActions(action).to_categorical_index()]
        # if self.action[0] == 0:
        # print("self action 0")

    def is_ball_going_towards_enemy(self):
        return self.ball_t[1] > self.ball_tp1[1]

    def is_ball_on_field(self):
        return self.ball_t[1] != 255 or self.ball_tp1[1] != 255

    def save(self):
        with open(OBS_PATH, "a") as outfile_X:
            np.savetxt(outfile_X, delimiter=",", X=[self.obs.flatten()], fmt="%d")
        with open(DIFF_PATH, "a") as outfile_X:
            state_before_copy = self.obs.copy()
            state_before_copy[:, -1] = 0
            diff = self.obs_tp1 - state_before_copy

            np.savetxt(outfile_X, delimiter=",", X=[diff.flatten()], fmt="%d")
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
        return ((obs[34:194:4, 40:142:2, 2] > 50).astype(np.uint8)).astype(float)


class GameObservations:
    def __init__(self):
        self.observations = []
        self.going_to_enemy = True
        self.should_save = False
        self.shouldSkipFrame = False

    def add_observation(self, obs_t, obs_tp1, action, reward):
        # print("observation added")
        if reward < 0:
            self.observations = []
            self.shouldSkipFrame = True
            return
        if reward > 0:
            for obs in self.observations:
                obs.save()
            self.observations = []
            self.shouldSkipFrame = True
            return
        observation = Observation(obs_t, obs_tp1)
        observation.add_action(action)
        if not observation.is_ball_on_field():
            # print("ball is not on field")
            return
        # else:
        #     print("")

        self.observations.append(observation)
        if observation.is_ball_going_towards_enemy() and not self.going_to_enemy:
            self.going_to_enemy = True
            if self.shouldSkipFrame:
                self.shouldSkipFrame = False
                self.observations = []

            # print("observation len", len(self.observations))
            for obs in self.observations:
                obs.save()
            self.observations = []
        elif not observation.is_ball_going_towards_enemy() and self.going_to_enemy:
            self.going_to_enemy = False
            self.observations = []


game_observations = GameObservations()


def callback(obs_t, obs_tp1, action, reward, *_):
    # print("action", action)
    if type(obs_t) is not np.ndarray:
        return
    game_observations.add_observation(obs_t, obs_tp1, action, reward)
    # observation = Observation(obs_t, obs_tp1, action, reward)
    # observation.add_action(action)
    # observation.save()


# try:
#     # with open(OBS_PATH, "a") as outfile_X:
#     #     np.savetxt(outfile_X, delimiter=",", X="obs", fmt="%d")
#     # with open(X_PATH, "a") as outfile_X:
#     #     np.savetxt(outfile_X, delimiter=",", X="x0,x1,x2,x3,x4,x5,x6,x7", fmt="%d")
#     # with open(Y_PATH, "a") as outfile_Y:
#     #     np.savetxt(outfile_Y, delimiter=",", X="y", fmt="%d")
#     # os.remove(X_PATH)
#     # os.remove(OBS_PATH)
#     # os.remove(Y_PATH)
# except Exception:
#     pass

ale.loadROM(Pong)
env = gym.make(
    "ALE/Pong-v5",
    render_mode="rgb_array",
)
env.reset()
play(
    env,
    zoom=3,
    fps=12,
    callback=callback,
    # keys_to_action={"z": 2, "d": 3, "k": 2, "j": 3},
)
env.close()
