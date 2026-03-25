from src.vision import WalkieVision
import cv2
import time

time.sleep(2)
walkie_vision = WalkieVision(camera_device=0, detection_provider="yolo", preload=True)

while True:
    image = walkie_vision.capture()
    cv2.imshow("Image", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

