import cv2

# camera = cv2.VideoCapture('http://Cartensz-PC.local:8000/camera/mjpeg?type=.mjpg')
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()
