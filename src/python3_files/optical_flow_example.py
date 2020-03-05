import cv2
import numpy as np
import pickle
#cap = cv2.VideoCapture("../glasses_reflection.mov")
cap = cv2.VideoCapture("sidewalk.m4v")
#vid = pickle.load(open("particle_vid.p", "rb"))

ret, frame1 = cap.read()
#print(frame1.shape)

#frame1 = vid[0]

prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

mags = []
angs = []

for i in range(50):
    print(i)

#    frame2 = vid[i]

    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

#    print(flow.shape)
#    print(flow)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

#    print(mag[0][0])
#    print(ang[0][0])

    mags.append(mag)
    angs.append(ang)

    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    cv2.imshow('frame2',rgb)
 #   k = cv2.waitKey(30) & 0xff
 #   if k == 27:
 #       break
 #   elif k == ord('s'):
 #       cv2.imwrite('opticalfb.png',frame2)
 #       cv2.imwrite('opticalhsv.png',rgb)
    prvs = next

pickle.dump((mags, angs), open("sidewalk.p", "wb"))

cap.release()
cv2.destroyAllWindows()