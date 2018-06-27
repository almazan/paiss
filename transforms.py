from PIL import Image
import numpy as np
import math
import torch

def normalize(im, norm):
    # Normalize
    for t, m, s in zip(im, norm['rgb_means'], norm['std']):
        t.sub_(m).div_(s)
    return im

def to_tensor(im):
    # Convert from PIL HWC to tensor CHW
    im = np.array(im)
    I = torch.from_numpy(im.transpose((2, 0, 1)))
    I = I.float().div(255)
    return I

def flip_image(im, flip):
    if flip:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    return im

def resize_image(im, S):
    # Resizes the image so the largest side is equal to S
    w, h = im.size

    new_ratio = float(S)/np.max((w,h))
    new_w = int(w * new_ratio)
    new_h = int(h * new_ratio)

    return im.resize((new_w, new_h), Image.BILINEAR)

def rotate(image, rotation):
    x = image.size[0]
    y = image.size[1]

    # Rotate, while expanding the canvas size
    image = image.rotate(rotation, expand=True, resample=Image.BICUBIC)

    # Get size after rotation, which includes the empty space
    X = image.size[0]
    Y = image.size[1]

    # Get our two angles needed for the calculation of the largest area
    angle_a = abs(rotation)
    angle_b = 90 - angle_a

    # Python deals in radians so get our radians
    angle_a_rad = math.radians(angle_a)
    angle_b_rad = math.radians(angle_b)

    # Calculate the sins
    angle_a_sin = math.sin(angle_a_rad)
    angle_b_sin = math.sin(angle_b_rad)

    # Find the maximum area of the rectangle that could be cropped
    E = (math.sin(angle_a_rad)) / (math.sin(angle_b_rad)) * \
        (Y - X * (math.sin(angle_a_rad) / math.sin(angle_b_rad)))
    E = E / 1 - (math.sin(angle_a_rad) ** 2 / math.sin(angle_b_rad) ** 2)
    B = X - E
    A = (math.sin(angle_a_rad) / math.sin(angle_b_rad)) * B

    # Crop this area from the rotated image
    # image = image.crop((E, A, X - E, Y - A))
    image = image.crop((int(round(E)), int(round(A)), int(round(X - E)), int(round(Y - A))))

    # Return the image, re-sized to the size of the image passed originally
    return image.resize((x, y), resample=Image.BICUBIC)

