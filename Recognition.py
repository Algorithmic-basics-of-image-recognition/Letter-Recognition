import cv2
import numpy as np
import joblib
import imutils
from skimage.feature import hog

classiferPath = "./letters.pkl"
image = "./example1.jpg"

clf, pp = joblib.load(classiferPath)

img = cv2.imread(image)
img = imutils.resize(img, height = 1000)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray, 5)
thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,8)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
dilate = cv2.dilate(thresh, kernel, iterations=10)

cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

for c in cnts:
    area = cv2.contourArea(c)
    if area > 1000:
        x,y,w,h = cv2.boundingRect(c)
        ROI = img[y:y+h, x:x+w]
        leng = int(h * 1.6)
        pt1 = int(y + h // 2 - leng // 2)
        pt2 = int(x + w // 2 - leng // 2)
        roi = img[pt1:pt1 + leng, pt2:pt2 + leng]
        # Resize the image
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = roi[:, ::-1, :]
        roi = np.rot90(roi, axes=(0, 1))
        roi = cv2.dilate(roi, (3, 3))
        # Calculate the HOG features
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1))
        roi_hog_fd = pp.transform(np.array([roi_hog_fd], 'float64'))
        nbr = clf.predict(roi_hog_fd)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(img, str(int(nbr[0])), (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

cv2.namedWindow("Resulting Image with Rectangular ROIs", cv2.WINDOW_NORMAL)
cv2.imshow("Resulting Image with Rectangular ROIs", img)
cv2.waitKey()
