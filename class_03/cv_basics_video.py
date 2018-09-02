import cv2

cap = cv2.VideoCapture(0)

while True:
	ret,frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	if ret == False:
		continue
	
	cv2.imshow('video', frame)
	cv2.imshow('gray', gray)


	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()