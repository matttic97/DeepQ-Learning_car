import numpy as np
import math
from pygame import gfxdraw

from utils import rotate_center, crop_view, ray_cast, get_ray_indices


class Car:
    def __init__(self, x, y, img, angle=270):
        self.x = x
        self.y = y
        self.img = img
        self.h = np.sqrt((img.get_width()//2)**2 + (img.get_height()//2)**2)
        self.angle = angle
        self.alpha = np.arcsin((self.img.get_width()//2) / self.h) * (180/np.pi)
        self.angle_power = 0
        self.angle_max_power = 5
        self.acceleration = 0.12
        self.friction = 0.08
        self.break_power = 0.22
        self.speed = 0
        self.max_speed = 7
        self.view_padding = (
            int(self.img.get_height()*3), int(self.img.get_height()*2),        # top, bottom
            int(self.img.get_height()*2), int(self.img.get_height()*2)          # left, right
        )
        self.setup_rays()

    def setup_rays(self):
        center = (self.view_padding[0], self.view_padding[2])
        self.ray0 = get_ray_indices(center, (119, 0))
        self.ray1 = get_ray_indices(center, (90, 0))
        self.ray2 = get_ray_indices(center, (45, 0))  # (0, 60)
        self.ray3 = get_ray_indices(center, (0, 13))  # (0, 25)
        self.ray4 = get_ray_indices(center, (0, 60))  # (60, 0)
        self.ray5 = get_ray_indices(center, (0, 107))  # (119, 25)
        self.ray6 = get_ray_indices(center, (45, 119))  # (119, 60)
        self.ray7 = get_ray_indices(center, (90, 119))  # (119, 90)
        self.ray8 = get_ray_indices(center, (119, 119))  # (119, 119)

    def adjust_steer_power(self):
        self.angle_power = self.angle_max_power * (1-math.exp(-2*abs(self.speed) / self.max_speed))

    def action(self, choice):
        self.adjust_steer_power()

        if choice in [1, 3, 4]:
            self.accelerate()
        elif choice in [2, 5, 6]:
            self.hit_break()
        else:
            self.decelerate()

        if choice in [3, 5, 7]:
            self.turn(1)
        elif choice in [4, 6, 8]:
            self.turn(-1)

        self.update()

    def accelerate(self):
        self.speed += self.acceleration
        self.speed = min(self.speed, self.max_speed)

    def decelerate(self):
        self.speed -= self.friction
        self.speed = max(self.speed, 0)

    def hit_break(self):
        self.speed -= self.break_power
        self.speed = max(self.speed, 0)

    def update(self):
        radians = math.radians(self.angle)
        vertical = math.cos(radians) * self.speed
        horizontal = math.sin(radians) * self.speed

        self.x -= horizontal
        self.y -= vertical

    def turn(self, direction):
        if abs(self.speed) > 0:
            self.angle += direction * np.sign(self.speed) * self.angle_power

    def draw(self, win):
        img, top_left = rotate_center(self.img, (self.x, self.y), self.angle)
        win.blit(img, top_left)
        loc = self.get_center()
        gfxdraw.pixel(win, loc[0], loc[1], [255, 0, 0])

    def get_center(self):
        return int(self.x + self.img.get_width()//2), int(self.y + self.img.get_height()//2)

    def get_state(self, view):
        ray0 = ray_cast(self.ray0, view)
        ray1 = ray_cast(self.ray1, view)
        ray2 = ray_cast(self.ray2, view)
        ray3 = ray_cast(self.ray3, view)
        ray4 = ray_cast(self.ray4, view)
        ray5 = ray_cast(self.ray5, view)
        ray6 = ray_cast(self.ray6, view)
        ray7 = ray_cast(self.ray7, view)
        ray8 = ray_cast(self.ray8, view)
        return np.array([ray0, ray1, ray2, ray3, ray4, ray5, ray6, ray7, ray8, self.speed/self.max_speed])

    def get_view(self, track):
        view, collision_field = crop_view(
            track,
            (self.get_center()),
            max(self.img.get_width(), self.img.get_height())*6,
            -self.angle,
            self.view_padding,
            (self.img.get_width(), self.img.get_height())
        )
        return view, collision_field
