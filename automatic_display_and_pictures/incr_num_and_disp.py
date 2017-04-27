import pickle
import os
import sys

imageCounter = pickle.load(open("img_counter.p", "r"))
IMAGE_BASE = "input_images/redgreen_"
EXTENSION = ".png"

commandBeginning = "open -a Preview " + IMAGE_BASE
commandEnd = EXTENSION + " ; /usr/bin/osascript -e 'tell application \"Preview\"' -e \"activate\" -e 'tell application \"System Events\"' -e 'keystroke \"f\" using {control down, command down}' -e \"end tell\" -e \"end tell\""

command = commandBeginning + str(imageCounter) + commandEnd

os.system(command)

pickle.dump(imageCounter+1, open("img_counter.p", "w"))