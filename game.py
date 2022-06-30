import numpy as np
import pygame
import math
import time
import cv2
from utils import rotate_center, crop_view
from pygame import gfxdraw


def action_from_keys(keys):
    action = 0
    if keys[pygame.K_UP] and keys[pygame.K_LEFT]:
        action = 3
    elif keys[pygame.K_UP] and keys[pygame.K_RIGHT]:
        action = 4
    elif keys[pygame.K_DOWN] and keys[pygame.K_LEFT]:
        action = 5
    elif keys[pygame.K_DOWN] and keys[pygame.K_RIGHT]:
        action = 6
    elif keys[pygame.K_UP]:
        action = 1
    elif keys[pygame.K_DOWN]:
        action = 2
    elif keys[pygame.K_LEFT]:
        action = 7
    elif keys[pygame.K_RIGHT]:
        action = 8
    return action


def check_bounds(player, window):
    pos = [int(player.x + player.img.get_width()//2), int(player.y + player.img.get_height()//2)]
    if pos[0] < 0 or pos[0] > window.get_width()\
            or pos[1] < 0 or pos[1] > window.get_height():
        player.speed = 0


class Car:
    def __init__(self, x, y, img):
        self.x = x
        self.y = y
        self.img = img
        self.h = np.sqrt((img.get_width()//2)**2 + (img.get_height()//2)**2)
        self.angle = 180
        self.alpha = np.arcsin((self.img.get_width()//2) / self.h) * (180/np.pi)
        self.angle_power = 0
        self.angle_max_power = 4
        self.acceleration = 0.2
        self.friction = 0.1
        self.speed = 0
        self.max_speed = 10

    def adjust_steer_power(self):
        self.angle_power = self.angle_max_power * (1-math.exp(-2*abs(self.speed) / self.max_speed))

    def action(self, choice):
        self.adjust_steer_power()

        if choice in [1, 3, 4]:
            self.accelerate(1)
        elif choice in [2, 5, 6]:
            self.accelerate(-1)
        else:
            self.decelerate()

        if choice in [3, 5, 7]:
            self.turn(1)
        elif choice in [4, 6, 8]:
            self.turn(-1)

    def accelerate(self, direction):
        self.speed += direction * self.acceleration
        if direction == 1:
            self.speed = min(self.speed, self.max_speed)
        else:
            self.speed = max(self.speed, -self.max_speed)
        self.update()

    def decelerate(self):
        if self.speed > 0:
            self.speed -= self.friction
        else:
            self.speed += self.friction

        if abs(self.speed) <= self.friction:
            self.speed = 0
        self.update()

    def update(self):
        radians = math.radians(self.angle)
        vertical = math.cos(radians) * self.speed
        horizontal = math.sin(radians) * self.speed

        self.x -= horizontal
        self.y -= vertical

    def turn(self, direction):
        if abs(self.speed) > 0:
            self.angle += direction * self.angle_power


    def draw(self, win):
        img, top_left = rotate_center(self.img, (self.x, self.y), self.angle)
        win.blit(img, top_left)
        #gfxdraw.pixel(win, int(self.x + self.img.get_width()//2), int(self.y + self.img.get_height()//2), [255, 0, 0])

    def get_view(self, track):
        return crop_view(
            track,
            (int(self.x + self.img.get_width()//2), int(self.y + self.img.get_height()//2)),
            max(self.img.get_width(), self.img.get_height())*6,
            -self.angle,
            (int(self.img.get_height()*1.5), self.img.get_width()*3),
            (self.img.get_width(), self.img.get_height())
        )


TRACK = pygame.image.load("templates/track_with_road.png")
CAR = pygame.image.load("templates/car.png")
TRACK_ndp = np.array(pygame.surfarray.array3d(TRACK).swapaxes(0, 1))

pygame.init()
WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Racing game")

FPS = 60
run = True
clock = pygame.time.Clock()

player = Car(0, 0, CAR)

while run:
    clock.tick(FPS)

    window, car = player.get_view(TRACK_ndp)
    cv2.imshow("Test", window)

    # check bounds
    check_bounds(player, WIN)

    WIN.blit(TRACK, (0, 0))
    player.draw(WIN)
    pygame.display.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            break

    keys = pygame.key.get_pressed()
    player.action(action_from_keys(keys))

pygame.quit()
