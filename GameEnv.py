import cv2
import pygame
import numpy as np
from car import Car


CRASH_PENALTY = 1
MOVE_PENALTY = 1
MAX_STEPS = 2500
GAME_ROUNDS = 3


class RacingEnv:
    def __init__(self):
        track = pygame.image.load("templates/track_with_road.png")
        self.TRACK = pygame.image.load("templates/track_goals.png")
        self.CAR = pygame.image.load("templates/car.png")
        self.track_ndp = np.array(pygame.surfarray.array3d(track).swapaxes(0, 1))

        self.player = Car(490, 57, self.CAR, 89)
        view, _ = self.player.get_view(self.track_ndp)
        self.OBSERVATION_SPACE_SIZE = 10  # self.player.get_state(view, self.player.view_padding).shape
        self.ACTION_SPACE_SIZE = 10

        self.fps = 120
        self.width, self.height = self.TRACK.get_width(), self.TRACK.get_height()
        self.WIN = pygame.display.set_mode((self.width, self.height))
        pygame.init()
        self.font = pygame.font.Font(pygame.font.get_default_font(), 36)
        pygame.display.set_caption("Racing game")

        self.reset()

    def reset(self):
        self.player = Car(425, 535, self.CAR, 270)

        self.goal_chain = GoalChain(1)
        self.rounds = 0

        self.episode_step = 0
        self.max_episode_steps = MAX_STEPS

        self.pause_counter = 0

        view, _ = self.player.get_view(self.track_ndp)
        state = self.player.get_state(view)

        return state

    def render(self):
        self.clock = pygame.time.Clock()
        self.WIN.blit(self.TRACK, (0, 0))
        self.player.draw(self.WIN)
        pygame.display.update()
        self.clock.tick(self.fps)

    def step(self, action, learn=False):
        self.episode_step += 1
        self.player.action(action)

        if action == 0:
            self.pause_counter += 1
        else:
            self.pause_counter = 0

        view, collision_field = self.player.get_view(self.track_ndp)
        new_state = self.player.get_state(view)

        # check for rewards
        done = False
        reward, round = self.goal_chain.step(self.player)
        if round:
            self.rounds += 1
            self.max_episode_steps += MAX_STEPS
            print('round')
            if self.rounds == GAME_ROUNDS:
                done = True
                print('win')
        elif np.sum(collision_field) > 0:
            reward = -CRASH_PENALTY
            done = True
            print('crash')
        # else:
        #     reward = -MOVE_PENALTY  # * (self.player.max_speed - self.player.speed) / self.player.max_speed)

        if learn and (self.episode_step >= self.max_episode_steps or self.pause_counter > 100):
            done = True
            print('timeout')

        if done:
            new_state = np.ones(10)*np.nan

        return new_state, reward, done


class Goal:
    def __init__(self, x1, x2, y1, y2, reward, start=False):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.reward = reward
        self.start = start
        self.achieved = False

    def reset(self):
        self.achieved = False

    def is_achieved(self, player):
        pos = player.get_center()
        if self.x1 <= pos[0] <= self.x2 and self.y1 <= pos[1] <= self.y2:
            self.achieved = True
        return self.achieved

    def get_reward(self, previous):
        if previous.achieved:
            previous.reset()
        return self.reward


class GoalChain:
    def __init__(self, next=1, normal_reward = 1, round_reward=1):
        self.goals = dict()
        self.next = next
        self.normal_reward = normal_reward
        self.round_reward = round_reward
        self.reset()

    def reset(self):
        self.goals = dict()
        self.goals[0] = Goal(357, 367, 493, 568, self.round_reward, True)
        self.goals[1] = Goal(462, 472, 494, 568, self.normal_reward)
        self.goals[2] = Goal(580, 590, 496, 568, self.normal_reward)
        self.goals[3] = Goal(780, 790, 496, 566, self.normal_reward)
        self.goals[4] = Goal(897, 983, 425, 435, self.normal_reward)
        self.goals[5] = Goal(907, 983, 216, 226, self.normal_reward)
        self.goals[6] = Goal(795, 805, 18, 89, self.normal_reward)
        self.goals[7] = Goal(494, 504, 29, 95, self.normal_reward)
        self.goals[8] = Goal(535, 545, 162, 217, self.normal_reward)
        self.goals[9] = Goal(546, 556, 305, 365, self.normal_reward)
        self.goals[10] = Goal(163, 173, 185, 334, self.normal_reward)
        self.goals[11] = Goal(262, 272, 489, 568, self.normal_reward)

    def get_previous(self, current):
        if current == 0:
            index = len(self.goals) - 1
        else:
            index = current - 1
        return self.goals[index]

    def update(self):
        if self.next == len(self.goals)-1:
            self.next = 0
        else:
            self.next += 1

    def step(self, player):
        reward = 0
        round = False
        if self.goals[self.next].is_achieved(player):
            round = self.goals[self.next].start
            reward = self.goals[self.next].get_reward(self.get_previous(self.next))
            self.update()
            print('jee')
        return reward, round
