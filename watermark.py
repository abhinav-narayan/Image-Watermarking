# -*- coding: utf-8 -*-
"""
Image Watermarking using Discrete Fourier Transform

"""

import cv2
import numpy as np
import matplotlib.pylab as plt
from separability import *
from skimage.io import imsave
from skimage.metrics import peak_signal_noise_ratio as psnr

orig_img = cv2.imread('C:/Users/Abhi/Downloads/lena.png')
watermark_img = cv2.imread('C:/Users/Abhi/Downloads/monkey.png')

orig_gray = cv2.cvtColor(orig_img,cv2.COLOR_BGR2GRAY)
watermark_gray = cv2.cvtColor(watermark_img,cv2.COLOR_BGR2GRAY)

orig_fft = separability_2ddft(orig_gray)
watermark_fft = separability_2ddft(watermark_gray)

p = np.angle(orig_fft)
p = np.exp(1j * p)

fused_img = np.abs(watermark_fft) * p
fused_ifft = np.uint8(np.fft.ifft2(fused_img))
p = psnr(orig_gray, fused_ifft)

cv2.imwrite('Watermarked_output_2.png',fused_ifft)