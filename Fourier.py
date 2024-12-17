import cv2
import numpy as np

# 讀取彩色影像
image = cv2.imread('more_draw.png', cv2.IMREAD_COLOR)

# 顯示原始影像
cv2.imshow("Original Image", image)

# 分離 RGB 通道
b, g, r = cv2.split(image)

# 函數來對單一通道進行傅立葉變換和濾波
def apply_fourier_transform(channel):
    # 傅立葉變換
    f = np.fft.fft2(channel)
    fshift = np.fft.fftshift(f)

    # 濾波處理：設置低通濾波器
    rows, cols = channel.shape[:2]
    crow, ccol = rows // 2, cols // 2

    # 創建低通濾波器
    mask = np.zeros((rows, cols), np.uint8)
    r = 30  # 濾波器的半徑
    cv2.circle(mask, (ccol, crow), r, 1, -1)

    # 應用濾波器
    fshift = fshift * mask

    # 逆傅立葉變換
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back.astype(np.uint8)

# 對每個通道進行傅立葉變換和濾波
r_back = apply_fourier_transform(r)
g_back = apply_fourier_transform(g)
b_back = apply_fourier_transform(b)

# 合併三個通道
img_back = cv2.merge([b_back, g_back, r_back])

# 顯示處理後的影像
cv2.imshow("Filtered Image (Fourier)", img_back)

denoised = cv2.fastNlMeansDenoisingColored(img_back, None, 10, 10, 7, 21)
cv2.imshow("Denoised", denoised)
cv2.waitKey(0)
# 等待用戶按任意鍵關閉視窗
cv2.destroyAllWindows()
