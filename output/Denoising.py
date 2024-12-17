import cv2
# 載入圖像
image = cv2.imread("../clothing_more.jpg")
# 應用高斯模糊
blurred = cv2.GaussianBlur(image, (11, 11), 0)  # 核大小為 (5, 5)，自動計算 sigma
# 顯示結果
cv2.imshow("Original", image)
cv2.imshow("Gaussian Blur", blurred)
denoised = cv2.fastNlMeansDenoisingColored(blurred, None, 10, 10, 7, 21)
cv2.imshow("Denoised", denoised)
cv2.waitKey(0)

