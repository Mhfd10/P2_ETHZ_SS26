import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from uncertainties.umath import cos


def plot_image_profiles(x,y, raw_data, title="Raw intensity profiles"):
    plt.figure(figsize=(8, 5))
    plt.style.use('default')
    plt.plot(x, raw_data, label="Raw data")
    plt.plot(x, y, label='filtered')
    plt.xlabel("Pixel number")
    plt.ylabel("Intensity (normalized)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(title + ".png", dpi=600)
    plt.show()


# registration and preprocessing of data

# reads in all tif files in a folder and returns a numpyarray in which every image is stored
def load_images(path):
    images = []

    for filename in os.listdir(path):
        if filename.endswith('.tif'):
            filepath = os.path.join(path, filename)
            image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            images.append(image)

    images = np.array(images)

    return images

# This part takes for each image in the images array a horizontal bandwidth, takes the average  of said bandwidth and writes it into a new array
def convert_into_array(images):
    band_width = 50 # adjust
    vline_height = 600 # adjust

    images_one_line = []
    raw_one_line = []

    for image in images:
        gray_raw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # noise filtering
        sigma = 4
        gray = cv2.GaussianBlur(gray_raw, (0, 0), sigmaX=sigma)

        height, width = gray.shape
        band_start = max(0, vline_height-band_width // 2)
        band_end   = min(height, vline_height+band_width // 2)

        # band extrahieren
        band = gray[band_start:band_end,:]
        band_raw = gray_raw[band_start:band_end,:]

        # Average pixel values across the horizontal band (to reduce band to one profile)
        values = band.mean(axis=0)
        values_raw = band_raw.mean(axis=0)

        # Normalize values to range [0, 1]
        values = (values - np.min(values)) / (np.max(values) - np.min(values))
        values_raw = (values_raw - np.min(values_raw)) / (np.max(values_raw) - np.min(values_raw))

        raw_one_line.append(values_raw)
        images_one_line.append(values)

    images_one_line = np.array(images_one_line)
    raw_one_line = np.array(raw_one_line)

    return images_one_line, raw_one_line

# This part does organisation of the data processed so far
def sort_data(path_images, path_notes, images_one_line, step):
    # collect and clean the filenames
    filenames = sorted([f for f in os.listdir(path_images) if f.lower().endswith(".tif")])
    names = [os.path.splitext(f)[0] for f in filenames]


    k_values = np.loadtxt(f'{path_notes}/{step}.txt', delimiter=',')
    k_values = k_values[:, 1]

    # build a dict keyed by image name
    img_dic = {
        name: {
            "k_value": float(k),
            "pixel_values": pix # 1-D array
        }
        for name, k, pix in zip(names, k_values, images_one_line)
    }

    return img_dic
