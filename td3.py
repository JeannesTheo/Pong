# @author:
# Sami El Kateb
# Theo Jeannes

import argparse
import gymnasium as gym
import os
import sys
from ale_py import ALEInterface
from ale_py.roms import Pong
from gymnasium.utils.play import play
from keras.src.saving.saving_api import load_model
from matplotlib.pyplot import Enum, np


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


X_PATH = "data/X.csv"
Y_PATH = "data/y.csv"
L_PLAYER_POS = 0
R_PLAYER_POS = 50
UNK_POSITION = np.array([-1, -1])


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
        with open(Y_PATH, "a") as outfile_Y:
            np.savetxt(outfile_Y, delimiter=",", X=self.action, fmt="%d")

    @staticmethod
    def __crop__(obs):
        # On coupe l'image pour ne garder que la partie intéressante du jeu,
        # sans le score, la raquette de l'ennemi et les bandes sur les cotés de l'écran
        return ((obs[34:194:4, 40:142:2, 2] > 50).astype(np.uint8)).astype(float)


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
    state_before = env.reset()[0]
    state_before = Observation.__crop__(state_before).reshape(-1, 40, 51, 1)
    state = None
    env.render()
    while True:
        if state is None or state_before is None:
            action = env.action_space.sample()
            state = state_before
            state_before, *_ = env.step(action)
            state_before = (Observation.__crop__(state_before)).reshape(-1, 40, 51, 1)
            continue

        state[0][:, -1] = 0
        state = state_before - state

        action = model.predict(state, verbose=False)
        action = np.argmax(action)
        action = PongActions.from_sparse_categorical(action)

        state = state_before
        state_before, *_ = env.step(action)
        state_before = (Observation.__crop__(state_before)).reshape(-1, 40, 51, 1)


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
        The mode of the python script. The play mode is for generating data to train the agent.
        The watch mode is for watching the agent play
        """,
    )
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    ale = ALEInterface()
    ale.loadROM(Pong)

    print(f"Starting the script in {args.mode} mode ...")
    if args.mode == "watch":
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'models', 'pong.h5')
        play_model(model_path)
    else:
        register_inputs()
