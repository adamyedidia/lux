import numpy as np
import cv2


def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 1)
        print((type(img[0][0][0])))

        reversedImage = (255*np.ones(img.shape) - img).astype(np.uint8)
#        print(img)
#        print(reversedImage)
        cv2.imshow('my webcam', reversedImage)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()


def main():
    show_webcam(mirror=True)


if __name__ == '__main__':
    main()
#@tonyfrost007