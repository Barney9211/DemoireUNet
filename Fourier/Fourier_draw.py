import cv2
import numpy as np
import matplotlib.pyplot as plt

# 讀取影像（灰度圖）
image = cv2.imread('../more.jpg', cv2.IMREAD_GRAYSCALE)

# 選取影像中某一行像素（中間行）
row = image[image.shape[0] // 2, :]

# 對該行像素進行一維傅立葉變換
f = np.fft.fft(row)
frequencies = np.fft.fftfreq(len(row))
magnitude = np.abs(f)

# 繪製波形（像素值）和頻譜圖
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(row)
plt.title("Pixel Intensity (Time Domain)")
plt.xlabel("Pixel Index")
plt.ylabel("Intensity")

plt.subplot(1, 2, 2)
plt.plot(frequencies[:len(frequencies) // 2], magnitude[:len(magnitude) // 2])
plt.title("Frequency Domain Spectrum")
plt.xlabel("Frequency")
plt.ylabel("Magnitude")

plt.tight_layout()
plt.show()
