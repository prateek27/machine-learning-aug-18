import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

cap = cv2.VideoCapture(0)

while True:
	ret,frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	if ret == False:
		continue

	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for (x,y,w,h) in faces:
		#cv2.putText(frame, 'vasu', (x,y-10), 1,(255,0,0),2,cv2.LINE_AA)
		cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

	cv2.imshow('face', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

