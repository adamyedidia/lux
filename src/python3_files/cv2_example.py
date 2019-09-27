
import cv2
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX
# mouse callback function
def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        i = 0
        while True:
            cv2.imshow('image',img) # to display the characters
            k = cv2.waitKey(0)
            cv2.putText(img, str((x,y)) , (x+i,y), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            i+=10
            # Press q to stop writing
            if k == ord('q'):
                break

def recordFourClicks(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        i = 0
        ret_val, img = cam.read()
#        cv2.imshow('my webcam', img)
            # to display the characters
#            k = cv2.waitKey(0)
        print((x, y))
#            cv2.putText(img, str((x,y)) , (x+i,y), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
 #           i+=10
            # Press q to stop writing
 #           if k == ord('q'):
 #               break

#        if cv2.waitKey(1) == 27: 
#            break  # esc to quit
    

# Create a black image, a window and bind the function to window

cam = cv2.VideoCapture(0)

for _ in range(100):
    ret_val, img = cam.read()


#img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',recordFourClicks)

while True:
    cv2.imshow('image',img)
    if cv2.waitKey(20) == 27:
        break
cv2.destroyAllWindows()
