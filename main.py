import pickle as pickle
import pygame
import GameEnv
from DQNAgent import DQNAgent
from tqdm import tqdm
import numpy as np
import time
import argparse


def action_from_keys(keys):
    action = 0
    if keys[pygame.K_SPACE] and keys[pygame.K_UP]:
        action = 0
    elif keys[pygame.K_SPACE] and keys[pygame.K_LEFT]:
        action = 5
    elif keys[pygame.K_SPACE] and keys[pygame.K_RIGHT]:
        action = 6
    elif keys[pygame.K_UP] and keys[pygame.K_LEFT]:
        action = 3
    elif keys[pygame.K_UP] and keys[pygame.K_RIGHT]:
        action = 4
    elif keys[pygame.K_UP]:
        action = 1
    elif keys[pygame.K_SPACE]:
        action = 2
    elif keys[pygame.K_LEFT]:
        action = 7
    elif keys[pygame.K_RIGHT]:
        action = 8
    return action


def drive(drive_manual=False):
    game = GameEnv.RacingEnv()
    game.fps = 60

    REPLAY_MEMORY_SIZE = 50_000
    MIN_REPLAY_MEMORY_SIZE = 2_000
    MODEL_NAME = "DENSEx256"
    MINIBATCH_SIZE = 256
    DISCOUNT = 0.98
    UPDATE_TARGET_EVERY = 10

    # Environment settings
    EPISODES = 20_000

    # Exploration settings
    epsilon = 1  # not a constant, going to be decayed
    EPSILON_DECAY = 0.9999975
    MIN_EPSILON = 0.01

    #  Stats settings
    AGGREGATE_STATS_EVERY = 50  # episodes

    agent = DQNAgent(DISCOUNT, game.ACTION_SPACE_SIZE, epsilon, MINIBATCH_SIZE, game.OBSERVATION_SPACE_SIZE,
                     MIN_REPLAY_MEMORY_SIZE, REPLAY_MEMORY_SIZE, UPDATE_TARGET_EVERY, EPSILON_DECAY, MIN_EPSILON,
                     'pretrained/best.model', 'pretrained/replay_history_h.pickle')

    for e in range(30):

        done = False

        current_state = game.reset()
        episode_reward = 0

        while not done:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break

            keys = pygame.key.get_pressed()
            action = action_from_keys(keys)
            #action = agent.get_max_q(current_state)
            print(current_state)
            new_state, reward, done = game.step(action)

            episode_reward += reward
            #print(episode_reward)

            agent.update_replay_memory((current_state, action, reward, new_state, done))
            current_state = new_state

            game.render()

    #     agent.learn(True)
    #
    # with open('models/replay_history_h.pickle', 'wb') as f:
    #     pickle.dump(agent.replay_memory, f, protocol=pickle.HIGHEST_PROTOCOL)
    # agent.model.save(
    #     f'models/base_new.model')
    pygame.quit()


def run_agent():

    REPLAY_MEMORY_SIZE = 50_000
    MIN_REPLAY_MEMORY_SIZE = 2_000
    MODEL_NAME = "DENSEx256"
    MINIBATCH_SIZE = 512
    DISCOUNT = 0.98
    UPDATE_TARGET_EVERY = 10

    # Environment settings
    EPISODES = 20_000

    # Exploration settings
    epsilon = 0.65  # not a constant, going to be decayed
    EPSILON_DECAY = 0.999997
    MIN_EPSILON = 0.01

    #  Stats settings
    AGGREGATE_STATS_EVERY = 10  # episodes
    SHOW_PREVIEW = False

    agent = DQNAgent(DISCOUNT, game.ACTION_SPACE_SIZE, epsilon, MINIBATCH_SIZE, game.OBSERVATION_SPACE_SIZE,
                     MIN_REPLAY_MEMORY_SIZE, REPLAY_MEMORY_SIZE, UPDATE_TARGET_EVERY, EPSILON_DECAY, MIN_EPSILON,
                     '/kaggle/input/models/base_new.model', '/kaggle/input/models/replay_history_h.pickle')

    # test
    agent.model.save(
        f'/kaggle/working/base_test.model')

    MAX_SCORE = -500

    ep_rewards = [-200]

    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        # Reset environment and get initial state
        current_state = game.reset()

        # Reset flag and start iterating until episode ends
        done = False
        print("\nEpsilon:", agent.epsilon)
        while not done:

            if step % 100 == 0:
                print(step, game.player.get_center(), episode_reward)

            action = agent.choose_action(current_state)

            new_state, reward, done = game.step(action, True)

            # Transform new continous state to new discrete state and count reward
            episode_reward += reward

            # if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
                # game.render()

            # Every step we update replay memory and train main network
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.learn(done)

            current_state = new_state
            step += 1

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])

            # Save model, but only when min reward is greater or equal a set value
            if max_reward >= MAX_SCORE:
                MAX_SCORE = max_reward
                agent.model.save(
                    f'/kaggle/working/base_best_{MODEL_NAME}__{max_reward:_>7.2f}__{int(time.time())}.model')
            else:
                agent.model.save(
                    f'/kaggle/working/base_{MODEL_NAME}__{episode}_episode.model')

            # save replay memory
            with open('/kaggle/working/replay_history_h.pickle', 'wb') as f:
                pickle.dump(agent.replay_memory, f, protocol=pickle.HIGHEST_PROTOCOL)


parser = argparse.ArgumentParser(description='Car Runner Script')
parser.add_argument("--agent_path", help="Path to pretrained agent", required=False, action='store')
args = parser.parse_args()

drive(args.agent_path)
