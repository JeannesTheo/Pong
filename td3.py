# @author:
# Sami El Kateb
# Theo Jeannes

import argparse
import gymnasium as gym
import os
import sys
import cv2
from ale_py import ALEInterface
from ale_py.roms import Pong
from gymnasium.utils.play import play
from keras.src.saving.saving_api import load_model
from joblib import load
from matplotlib.pyplot import Enum, np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, 'data')
X_PATH = os.path.join(DATA_PATH, 'X.csv')
Y_PATH = os.path.join(DATA_PATH, 'y.csv')
NEURAL_NET_PATH = os.path.join(CURRENT_DIR, 'models', 'pong-smoteen.h5')
RANDOM_FOREST_PATH = os.path.join(CURRENT_DIR, 'models', 'pong-random-forest.joblib')
DEBUG_PATH = os.path.join(CURRENT_DIR, 'debug')
IS_DEBUG = False


class PongActions(Enum):
    NO_ACTION = 0
    UP = 2
    DOWN = 3

    @staticmethod
    def from_sparse_categorical(num):
        if num == 0:
            return PongActions.NO_ACTION.value
        if num == 1:
            return PongActions.UP.value
        if num == 2:
            return PongActions.DOWN.value
        raise ValueError("Invalid Pong Action Category")

    def to_sparse_categorical(self):
        if self == PongActions.NO_ACTION:
            return 0
        if self == PongActions.UP:
            return 1
        if self == PongActions.DOWN:
            return 2
        raise ValueError("Invalid Pong Action Category")


class Observation:
    # Pour obtenir une observation
    def __init__(self, obs_t, obs_tp1) -> None:
        self.action = None
        self.obs = Observation.__crop__(obs_t) # On decoupe l'image pour ne garder que la partie interessante,
        self.obs_tp1 = Observation.__crop__(obs_tp1) # en noir et blanc pour reduire les dimensions

        # On identifie la position de la balle sur l'image,
        # ce qui permet de choisir les images que l'on souhaite sauvegarder
        obs_t_ball_only = self.obs.copy()[:, :-1]
        ball_t = np.argwhere(obs_t_ball_only == 1)
        obs_tp1_ball_only = self.obs_tp1.copy()[:, :-1]
        ball_tp1 = np.argwhere(obs_tp1_ball_only == 1)

        self.is_ball_on_field = len(ball_t) > 0 or len(ball_tp1) > 0

        if len(ball_t) > 0 and len(ball_tp1) > 0:
            self.is_ball_going_towards_enemy = ball_t[0][1] > ball_tp1[0][1]
        else:
            self.is_ball_going_towards_enemy = False

    def add_action(self, action):
        self.action = [PongActions(action).to_sparse_categorical()]

    def save(self):
        with open(X_PATH, "a") as outfile_X: # On sauvegarde la difference entre l'etat actuel et l'etat suivant
            # pour avoir une indication sur la direction de la balle lors de la prediction
            state_before_copy = self.obs.copy()
            state_before_copy[:, -1] = 0  # Si la raquette ne bouge pas, la soustraction
            # des deux images la ferait disparaitre de l'observation
            # On met donc a zero la colonne correspondant a la raquette dans l'état actuel pour corriger ce problème
            diff = self.obs_tp1 - state_before_copy
            np.savetxt(outfile_X, delimiter=",", X=[diff.flatten()], fmt="%d")
            if IS_DEBUG:
                Observation.save_debug(diff)
        with open(Y_PATH, "a") as outfile_Y:
            np.savetxt(outfile_Y, delimiter=",", X=self.action, fmt="%d")

    @staticmethod
    def __crop__(obs):
        # On coupe l'image pour ne garder que la partie intéressante du jeu,
        # sans le score, la raquette de l'ennemi et les bandes sur les cotés de l'écran
        return ((obs[34:194:4, 40:142:2, 2] > 50).astype(np.uint8)).astype(float)

    debug_image_nb = 0

    @staticmethod
    def save_debug(input_obs):
        obs = input_obs if len(input_obs.shape) == 2 else input_obs[0, :, :, 0]
        rgb_image = np.zeros((obs.shape[0], obs.shape[1], 3), dtype=np.uint8)
        rgb_image[obs == -1, 0] = 255
        rgb_image[obs == 1, 2] = 255
        filename = os.path.join(DEBUG_PATH, f'image_{Observation.debug_image_nb}.png')
        cv2.imwrite(filename, rgb_image)
        Observation.debug_image_nb += 1

    @staticmethod
    def preprocess_obs(obs):
        return (Observation.__crop__(obs)).reshape(-1, 40, 51, 1)


class GameObservations:
    def __init__(self):
        self.observations = []
        self.ball_directed_toward_enemy = True
        self.should_save = False

    # Cette méthode permet l'ajout d'observations à la liste des observations,
    # ainsi que leur sauvegarde si les conditions nécessaires sont réunies.
    def add_observation(self, obs_t, obs_tp1, action, reward):
        if reward < 0:  # Si on perd la balle, on supprime les observations courantes,
            # puisque qu'on ne veut pas apprendre à perdre
            self.observations = []
            return

        if reward > 0:  # Si on gagne un point, on sauvegarde les observations, puis on remet a zero la liste
            for obs in self.observations:
                obs.save()
            self.observations = []
            return
        # On cree l'observation, qui est la difference entre l'etat actuel et l'etat suivant, et on l'ajoute a la
        # liste des observations
        observation = Observation(obs_t, obs_tp1)
        observation.add_action(
            action)  # Si la balle n'est pas affichée (notamment lorsqu'elle reapparait apres un point), on ne garde
        # pas l'observation
        if not observation.is_ball_on_field:
            return

        self.observations.append(observation)
        # Toutes les observations sont sauvegardés ou supprimées lorsque la balle change de direction.
        # On sauvegarde les observations de la balle arrivant de notre coté lorsque la balle repart vers l'ennemi,
        # et on supprime les observations de la balle qui part vers l'ennemi lorsque la balle vient vers nous
        # Cela permet de limiter le nombre d'etats ou notre raquette ne bouge pas
        if observation.is_ball_going_towards_enemy and not self.ball_directed_toward_enemy:  # Si la balle va vers l'ennemi,
            # on sauvegarde les observations, puis on remet a zero la liste
            self.ball_directed_toward_enemy = True
            for obs in self.observations:
                obs.save()
            self.observations = []
        elif not observation.is_ball_going_towards_enemy and self.ball_directed_toward_enemy:  # Si la balle va vers nous, on
            # supprime les observations, puis on remet a zero la liste
            self.ball_directed_toward_enemy = False
            self.observations = []


def register_inputs():
    game_observations = GameObservations()

    def callback(obs_t, obs_tp1, action, reward, *_):
        if type(obs_t) is not np.ndarray:
            return
        game_observations.add_observation(obs_t, obs_tp1, action, reward)

    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    env.reset()
    play(env, zoom=3, fps=12, callback=callback)
    env.close()


def play_neural_net():
    env = gym.make("ALE/Pong-v5", render_mode="human")
    model = load_model(NEURAL_NET_PATH)
    state_before = env.reset()[0]
    state_before = Observation.preprocess_obs(state_before)
    state = None
    env.render()
    while True:
        if state is None or state_before is None:
            action = env.action_space.sample()
            state = state_before
            state_before, *_ = env.step(action)
            state_before = Observation.preprocess_obs(state_before)
            continue

        state[0][:, -1] = 0
        state = state_before - state

        if IS_DEBUG:
            Observation.save_debug(state)

        action = model.predict(state, verbose=False)
        action = np.argmax(action)
        action = PongActions.from_sparse_categorical(action)

        state = state_before
        state_before, *_ = env.step(action)
        state_before = Observation.preprocess_obs(state_before)


def play_random_forest():
    env = gym.make("ALE/Pong-v5", render_mode="human")
    model = load(RANDOM_FOREST_PATH)
    state_before = env.reset()[0]
    state_before = Observation.preprocess_obs(state_before)
    state = None
    env.render()
    while True:
        if state is None or state_before is None:
            action = env.action_space.sample()
            state = state_before
            state_before, *_ = env.step(action)
            state_before = Observation.preprocess_obs(state_before)
            continue

        state[0][:, -1] = 0
        state = state_before - state

        if IS_DEBUG:
            Observation.save_debug(state)

        action = model.predict(state.reshape(-1, 40 * 51))
        action = PongActions.from_sparse_categorical(action)

        state = state_before
        state_before, *_ = env.step(action)
        state_before = Observation.preprocess_obs(state_before)


# register_inputs() permet d'ajouter des données pour permettre a l'agent de s'entrainer. Une fois l'entrainement
# fait, grace au Jupiter notebook, on peut fournir le modèle entrainé, qui sert d'oracle à l'agent, en utilisant
# play_model. Le parametre correspond au chemin vers le modèle
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Pong Game played by a Reinforcement Learning Agent.
                       Auteurs: EL KATEB Sami, JEANNES Theo"""
    )
    parser.add_argument(
        "mode",
        metavar="mode",
        type=str,
        choices=["play", "watch"],
        help="""
        Accepted values: play | watch.
        The mode of the python script.
        The play mode is for generating data to train the agent.
        The watch mode is for watching the agent play.
        """,
    )
    parser.add_argument('--agent', type=str, default="neural_net", choices=["neural_net", "random_forest"],
                        help="""The algorithm that train the model that the agent will use""")
    parser.add_argument('--debug', action='store_true', help="Will create images of the observation state in the debug folder.")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    ale = ALEInterface()
    ale.loadROM(Pong)

    IS_DEBUG = args.debug
    if IS_DEBUG:
        os.makedirs(DEBUG_PATH, exist_ok=True)

    if args.mode == "watch":
        print(f"Starting the script in {args.mode} mode with {args.agent} agent ...")
        # Les modeles ont beau avoir des prédictions correctes, l'agent ne joue pas correctement et a tendance a se bloquer dans un coin selon les configurations.
        # Augmenter le nombre de données d'entrainement pourrait ameliorer le problème, puisque cela permettrait d'entrainer le modele sur plus de situations
        if args.agent == "random_forest":
            play_random_forest()
        else:
            play_neural_net()
    else:
        print(f"Starting the script in {args.mode} mode ...")
        os.makedirs(DATA_PATH, exist_ok=True)
        register_inputs()
