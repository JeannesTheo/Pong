from gymnasium.utils.play import play
import gymnasium as gym
from ale_py import ALEInterface
from keras.src.saving.saving_api import load_model
from matplotlib.pyplot import Enum, np
from ale_py.roms import Pong


class PongActions(Enum):
    NO_ACTION = 0
    UP = 2
    DOWN = 3

    @staticmethod
    def from_categorical_index(num):
        if num == 0:
            return PongActions.NO_ACTION.value
        if num == 1:
            return PongActions.UP.value
        if num == 2:
            return PongActions.DOWN.value
        return -1

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
        self.action = None
        self.obs = Observation.__crop__(obs_t)
        self.obs_tp1 = Observation.__crop__(obs_tp1)
        obs_t_preprocessed = Observation.__preprocess__(obs_t)
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

        self.data = np.array([self.ball_t, self.ball_tp1, right_player])

    def add_action(self, action):
        self.action = [PongActions(action).to_categorical_index()]

    def is_ball_going_towards_enemy(self):
        return self.ball_t[1] > self.ball_tp1[1]

    def is_ball_on_field(self):
        return self.ball_t[1] != 255 or self.ball_tp1[1] != 255

    def save(self):
        with open(OBS_PATH, "a") as outfile_X:
            np.savetxt(outfile_X, delimiter=",", X=[self.obs.flatten()], fmt="%d")
        with open(DIFF_PATH, "a") as outfile_X:
            diff = self.obs - self.obs_tp1
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
        return (obs[34:194:4, 40:142:2, 2] > 50).astype(np.uint8)


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
            print("ball is not on field")
            return
        else:
            print("")

        self.observations.append(observation)
        if observation.is_ball_going_towards_enemy() and not self.going_to_enemy:
            self.going_to_enemy = True
            if self.shouldSkipFrame:
                self.shouldSkipFrame = False
                self.observations = []
            for obs in self.observations:
                obs.save()
            self.observations = []
        elif not observation.is_ball_going_towards_enemy() and self.going_to_enemy:
            self.going_to_enemy = False
            self.observations = []


def register_inputs():
    global game_observations
    game_observations = GameObservations()

    def callback(obs_t, obs_tp1, action, reward, *_):
        if type(obs_t) is not np.ndarray:
            return
        game_observations.add_observation(obs_t, obs_tp1, action, reward)

    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    env.reset()
    play(env, zoom=3, fps=12, callback=callback)
    env.close()


def play_model(name_model):
    env = gym.make("ALE/Pong-v5", render_mode="human")
    model = load_model(name_model)
    print("loaded")
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

        state = state - state_before

        action = model.predict(state)
        action = np.argmax(action)
        print(action)
        action = PongActions.from_categorical_index(action)

        state = state_before
        (state_before, reward, terminated, *rest) = env.step(action)
        state_before = (Observation.__crop__(state_before)).reshape(-1, 40, 51, 1)


# register_inputs() permet d'ajouter des données pour permettre a l'agent de s'entrainer. Une fois l'entrainement
# fait, grace au Jupiter notebook, on peut fournir le modèle entrainé, qui sert d'oracle à l'agent, en utilisant
# play_model. Le parametre correspond au chemin vers le modèle
if __name__ == "__main__":
    ale = ALEInterface()
    ale.loadROM(Pong)
    # register_inputs()
    play_model("./models/pong-6.h5")
