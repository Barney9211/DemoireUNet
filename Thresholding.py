import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
if image is None:
    print("image is not exist")
cv2.imshow("origin",image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def gamma_correction(f, gamma=2.0):
    c = 255.0 / (255.0 ** gamma)
    table = np.array([min(255, max(0, int(round(i ** gamma * c, 0)))) for i in range(256)], dtype=np.uint8)
    return cv2.LUT(f, table)
imageGamma = gamma_correction(image ,2)

cv2.imshow("gamma " ,imageGamma)


binaryGamma = cv2.adaptiveThreshold(imageGamma, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imshow("binary Gamma",binaryGamma)
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imshow("Binary", binary)
cv2.waitKey(0)

