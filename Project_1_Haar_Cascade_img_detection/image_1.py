import cv2

img_path = input("Image: ")
img = cv2.imread(img_path)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1, 20)

for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 3)

cv2.imshow("Sameer", img)
cv2.imwrite("Detected2.jpeg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()