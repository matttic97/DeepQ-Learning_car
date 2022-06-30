import pickle as pickle
import pygame
import GameEnv
from DQNAgent import DQNAgent
from ManualDriver import ManualDriver
from tqdm import tqdm
import time
import argparse


def drive(agent_path):
    game = GameEnv.RacingEnv()
    game.fps = 60

    if agent_path is not None:
        agent = DQNAgent(game.ACTION_SPACE_SIZE, game.OBSERVATION_SPACE_SIZE, agent_path='pretrained/best.model')
    else:
        agent = ManualDriver(game)

    for e in range(30):

        done = False

        current_state = game.reset()
        episode_reward = 0

        while not done:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break

            action = agent.next_action(current_state)

            new_state, reward, done = game.step(action)

            episode_reward += reward
            #print(episode_reward)

            current_state = new_state

            game.render()

        print(episode_reward)

    pygame.quit()


parser = argparse.ArgumentParser(description='Car Runner Script')
parser.add_argument("--agent_path", help="Path to pretrained agent", required=False, action='store')
args = parser.parse_args()

drive(args.agent_path)
