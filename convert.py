import cv2
import matplotlib.pyplot as plt

img = cv2.imread("yesol.jpg")   #내사진
img.shape

b, g, r = cv2.split(img)
img = cv2.merge([r,g,b])

img_resized = cv2.resize(img, dsize=(30 , 40)) #3:4 비율

plt.subplot(221)
plt.imshow(img, cmap=plt.cm.gray)
plt.title("DoDotYourSelf Upload Image")
plt.axis("off")

plt.subplot(222)
plt.imshow(img_resized, cmap=plt.cm.gray)
plt.title("DoDotYourSelf Change Image")
plt.axis("off")

plt.show()
