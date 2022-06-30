import pygame
import cv2
import numpy as np
from skimage.draw import line


def rotate_center(image, top_left, angle):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center=image.get_rect(topleft=top_left).center)
    return rotated_image, new_rect.topleft


def crop_view(image, loc, size, angle, win_padding, shape_car):
    x1, x2, y1, y2 = loc[0]-size//2, loc[0]+size//2, loc[1]-size//2, loc[1]+size//2
    bottom, top, left, right = 0, 0, 0, 0
    if x1 < 0:
        left = x1 * -1
        x1 = 0
    if x2 > image.shape[1]:
        right = x2-image.shape[1]
        x2 = image.shape[1]
    if y1 < 0:
        top = y1 * -1
        y1 = 0
    if y2 > image.shape[0]:
        bottom = y2-image.shape[0]
        y2 = image.shape[0]
    image = image[y1:y2, x1:x2]
    cropped = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    rotated = ModifiedWay(cropped, angle)
    cx, cy = rotated.shape[:2]

    window = rotated[cx//2 - win_padding[0]:cx//2 + win_padding[1], cy//2 - win_padding[2]:cy//2 + win_padding[3]]

    cx, cy = window.shape[:2]
    car = window[cx//2:cx//2 + shape_car[1], cy//2 - shape_car[0]//2:cy//2 + shape_car[0]//2]

    return cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)/255, cv2.cvtColor(car, cv2.COLOR_BGR2GRAY)/255


def ModifiedWay(rotateImage, angle):
    # Taking image height and width
    imgHeight, imgWidth = rotateImage.shape[0], rotateImage.shape[1]

    # Computing the centre x,y coordinates
    # of an image
    centreY, centreX = imgHeight // 2, imgWidth // 2

    # Computing 2D rotation Matrix to rotate an image
    rotationMatrix = cv2.getRotationMatrix2D((centreY, centreX), angle, 1.0)

    # Now will take out sin and cos values from rotationMatrix
    # Also used numpy absolute function to make positive value
    cosofRotationMatrix = np.abs(rotationMatrix[0][0])
    sinofRotationMatrix = np.abs(rotationMatrix[0][1])

    # Now will compute new height & width of
    # an image so that we can use it in
    # warpAffine function to prevent cropping of image sides
    newImageHeight = int((imgHeight * sinofRotationMatrix) +
                         (imgWidth * cosofRotationMatrix))
    newImageWidth = int((imgHeight * cosofRotationMatrix) +
                        (imgWidth * sinofRotationMatrix))

    # After computing the new height & width of an image
    # we also need to update the values of rotation matrix
    rotationMatrix[0][2] += (newImageWidth / 2) - centreX
    rotationMatrix[1][2] += (newImageHeight / 2) - centreY

    # Now, we will perform actual image rotation
    rotatingimage = cv2.warpAffine(
        rotateImage, rotationMatrix, (newImageWidth, newImageHeight))

    return rotatingimage


def get_ray_indices(start, end):
    discrete_line = list(zip(*line(*start, *end)))
    return discrete_line


def ray_cast(ray, image):
    casted = np.array([image[i] for i in ray])
    c = np.where(casted > 0)
    if len(c[0]) > 0:
        return min(1, 1 - ((c[0][0]-7) / 94))  # 1-close, 0-far
    return 0
