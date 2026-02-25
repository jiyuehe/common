import numpy as np
from PIL import Image # pip install pillow

def execute(image_file_name): # crop result images to remove excess whitespace
    img = Image.open(image_file_name).convert("RGBA")
    data = np.array(img)

    # true where pixel is not white
    threshold = 245  # define near-white as background
    mask = np.any(data[:, :, :3] < threshold, axis=2)
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    img.crop((x0, y0, x1, y1)).save(image_file_name)
