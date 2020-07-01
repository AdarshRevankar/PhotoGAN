import os
import time

from skimage.measure import compare_ssim
import cv2
import numpy as np
from PIL import Image

path_generated = 'eval/generated'
path_real = 'eval/real'


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return (1. - err / 65025.) * 100.


def ssim(A, B):
    (score, d) = compare_ssim(A, B, full=True, multichannel=True)
    # cv2.imshow("Title", d)
    # cv2.waitKeyEx(2000)
    return score


def load_image(folder):
    # Load the image to the list and return
    images = os.listdir(folder)
    img_lst = []
    opencv_img_lst = []
    for image in images:
        file_name = os.path.join(folder, image)
        img_lst.append(Image.open(file_name, "r").convert('LA'))
        opencv_img_lst.append(cv2.imread(file_name))
    return img_lst, opencv_img_lst


def resize(img):
    return np.array(img.resize((256, 256)))


def compute_score(real, gen):
    # Compute the score
    N = len(real)
    return ((255 - abs(real - gen)) / 255.).sum() / float(N * N * 3)


def evaluate():
    # loads image from real and generated image
    # Compares and returns the score
    real_img_lst, real_cv_img = load_image(path_real)
    gen_img_lst, gen_cv_img = load_image(path_generated)

    score = 0.
    N = len(real_img_lst)

    # For each image
    mse_lst, ssim_lst = [], []

    for real_img, gen_img in zip(real_img_lst, gen_img_lst):
        real_img_np, gen_img_np = resize(real_img), resize(gen_img)
        mse_lst.append(mse(real_img_np, gen_img_np))

    for real_img, gen_img in zip(real_cv_img, gen_cv_img):
        real_img = cv2.resize(real_img, (256, 256))
        gen_img = cv2.resize(gen_img, (256, 256))
        ssim_lst.append(ssim(real_img, gen_img))

    for m, s in zip(mse_lst, ssim_lst):
        print("MSE Accuracy : ", m, " ---- SSIM : ", round(s, 4))
    print(sum(mse_lst)/len(mse_lst), sum(ssim_lst)/len(ssim_lst))


evaluate()
