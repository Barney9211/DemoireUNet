import numpy as np
import argparse
import cv2

# 命令列參數解析
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# 讀取影像
img1 = cv2.imread(args["image"])
if img1 is None:
    print("Error: Image loading failed.")
    exit()

# Gamma 校正函數
def gamma_correction(f, gamma=2.0):
    c = 255.0 / (255.0 ** gamma)
    table = np.array([min(255, max(0, int(round(i ** gamma * c, 0)))) for i in range(256)], dtype=np.uint8)
    return cv2.LUT(f, table)  # 使用查找表進行運算

# 應用 Gamma 校正
img2 = gamma_correction(img1, 0.1)

# 顯示結果
cv2.imshow("Original", img1)
cv2.imshow("After", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
