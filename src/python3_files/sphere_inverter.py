from PIL import Image
import sys
from math import pi, sin, cos, floor, ceil

IMAGE_PATH = "bubble.jpg"

imageData = Image.open(IMAGE_PATH)
width, height = imageData.size
pixelData = imageData.load()

centerX = width/2.
centerY = height/2.

pixelRadius = min(centerX, centerY) # Hopefully they aren't too different!

# Ranges from 0 to 2pi
DIRECTION_ANGLE = float(sys.argv[1])

# Ranges from 0 to pi
DISTANCE_ANGLE = float(sys.argv[2])

# A small positive angle
DIRECTION_SIDE_LENGTH_RADS = float(sys.argv[3])

DISTANCE_SIDE_LENGTH_RADS = float(sys.argv[4])

PIXELS_PER_RADIAN = 1000.

dirSl = int(DIRECTION_SIDE_LENGTH_RADS*PIXELS_PER_RADIAN)
disSl = int(DISTANCE_SIDE_LENGTH_RADS*PIXELS_PER_RADIAN)

newImage = Image.new("RGB", (disSl, dirSl))

markedPixels = {}

def dot(list1, list2):
    return sum([i*j for i,j in zip(list1, list2)])

listOfPixels = []
# Build the list of pixels
for dirPixel in range(dirSl):
    littleAngleDirection = dirPixel / PIXELS_PER_RADIAN
    angleDirection = littleAngleDirection + DIRECTION_ANGLE

    print(str(dirPixel) + " / " + str(dirSl))

    for disPixel in range(disSl):
        littleAngleDistance = disPixel / PIXELS_PER_RADIAN
        angleDistance = littleAngleDistance + DISTANCE_ANGLE

        # This is in pixels
        distanceFromCenter = pixelRadius * sin(angleDistance / 2)

        distanceX = distanceFromCenter * cos(angleDirection)
        distanceY = distanceFromCenter * sin(angleDirection)

        x = centerX + distanceX
        y = centerY + distanceY

        lowerLeft = imageData.getpixel((floor(x), floor(y)))
        lowerRight = imageData.getpixel((ceil(x), floor(y)))
        upperLeft = imageData.getpixel((floor(x), ceil(y)))
        upperRight = imageData.getpixel((ceil(x), ceil(y)))

        rightWeight = x - floor(x)
        leftWeight = ceil(x) - x
        upperWeight = y - floor(y)
        lowerWeight = ceil(y) - y

        lowerLeftWeight = lowerWeight * leftWeight
        lowerRightWeight = lowerWeight * rightWeight
        upperLeftWeight = upperWeight * leftWeight
        upperRightWeight = upperWeight * rightWeight

        nearbyPixels = [lowerLeft, lowerRight, upperLeft, upperRight]
        listOfWeights = [lowerLeftWeight, lowerRightWeight, upperLeftWeight, \
            upperRightWeight]

        for coords in [(floor(x), floor(y)), (floor(x), ceil(y)), \
                        (ceil(x), floor(y)), (ceil(x), ceil(y))]:
            markedPixels[coords] = None

        pixelR = int(dot([i[0] for i in nearbyPixels], listOfWeights))
        pixelG = int(dot([i[1] for i in nearbyPixels], listOfWeights))
        pixelB = int(dot([i[2] for i in nearbyPixels], listOfWeights))

#        print pixelR, pixelG, pixelB

        listOfPixels.append((pixelR, pixelG, pixelB))

#for disPixel in range(disSl):
#    littleAngleDistance = disPixel / PIXELS_PER_RADIAN
#    angleDistance = littleAngleDistance + DISTANCE_ANGLE
#
#    # This is in pixels
#    distanceFromCenter = pixelRadius * sin(angleDistance / 2)
#
#    print str(disPixel) + " / " + str(disSl)
#
#    for dirPixel in range(dirSl):
#
#        littleAngleDirection = dirPixel / PIXELS_PER_RADIAN
#        angleDirection = littleAngleDirection + DIRECTION_ANGLE
#
#        distanceX = distanceFromCenter * cos(angleDirection)
#        distanceY = distanceFromCenter * sin(angleDirection)
#
#        x = centerX + distanceX
#        y = centerY + distanceY
#
#        lowerLeft = imageData.getpixel((floor(x), floor(y)))
#        lowerRight = imageData.getpixel((ceil(x), floor(y)))
#        upperLeft = imageData.getpixel((floor(x), ceil(y)))
#        upperRight = imageData.getpixel((ceil(x), ceil(y)))
#
#        rightWeight = x - floor(x)
#        leftWeight = ceil(x) - x
#        upperWeight = y - floor(y)
#        lowerWeight = ceil(y) - y
#
#        lowerLeftWeight = lowerWeight * leftWeight
#        lowerRightWeight = lowerWeight * rightWeight
#        upperLeftWeight = upperWeight * leftWeight
#        upperRightWeight = upperWeight * rightWeight
#
#        nearbyPixels = [lowerLeft, lowerRight, upperLeft, upperRight]
#        listOfWeights = [lowerLeftWeight, lowerRightWeight, upperLeftWeight, \
#            upperRightWeight]
#
#        for coords in [(floor(x), floor(y)), (floor(x), ceil(y)), \
#                        (ceil(x), floor(y)), (ceil(x), ceil(y))]:
#            markedPixels[coords] = None
#
#        pixelR = int(dot([i[0] for i in nearbyPixels], listOfWeights))
#        pixelG = int(dot([i[1] for i in nearbyPixels], listOfWeights))
#        pixelB = int(dot([i[2] for i in nearbyPixels], listOfWeights))
#
##        print pixelR, pixelG, pixelB
#
#        listOfPixels.append((pixelR, pixelG, pixelB))

for coords in markedPixels:
    pixelData[coords[0], coords[1]] = (0,0,0)
#        print listOfPixels

imageData.save("image_marked.png")
imageData.show()

newImage.putdata(listOfPixels)
newImage.save("flattened.png")
newImage.show()
