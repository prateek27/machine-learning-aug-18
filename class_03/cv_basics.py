import cv2

img = cv2.imread("download.jpeg")
gray = cv2.imread("download.jpeg", cv2.IMREAD_GRAYSCALE)


cv2.imshow('image', img)
cv2.imshow('grayscale', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()