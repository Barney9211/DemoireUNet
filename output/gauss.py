import cv2
# 載入圖像
image = cv2.imread("../clothing_more.jpg")

# 應用高斯模糊
blurred = cv2.GaussianBlur(image, (11, 11), 0)  # 核大小為 (5, 5)，自動計算 sigma

# 顯示結果
cv2.imshow("Original", image)
cv2.imshow("Gaussian Blur", blurred)

gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

# 計算拉普拉斯濾波
laplacian = cv2.Laplacian(gray, cv2.CV_64F)

# 將拉普拉斯濾波結果轉換為 uint8
laplacian_uint8 = cv2.convertScaleAbs(laplacian)

# 使用加權和進行銳化
sharpened = cv2.addWeighted(gray, 1.5, laplacian_uint8, -0.5, 0)

cv2.imshow("Sharpened", sharpened)

cv2.waitKey(0)

