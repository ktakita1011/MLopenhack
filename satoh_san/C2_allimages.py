from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import glob
get_ipython().run_line_magic('matplotlib', 'inline')


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr



files = glob.glob('./gear_images/*/*.jpeg')
dst_dir = './converted'
thumb_width = 128


for f in files:
    im = Image.open(f)

    # resize
    im_thumbnail = (expand2square(im, (255,255,255))).resize((thumb_width, thumb_width), Image.LANCZOS)
    arr = np.array(im_thumbnail)

    # normalize
    new_img = Image.fromarray(normalize(arr).astype('uint8'),'RGB')

    # save
    save_dir = os.path.join(dst_dir, os.path.basename(os.path.dirname(f)))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    new_img.save(os.path.join(save_dir, os.path.basename(f)), quality=95)

    