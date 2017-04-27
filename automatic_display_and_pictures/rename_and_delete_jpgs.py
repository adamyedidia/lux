import pickle
import os

imageCounter = pickle.load(open("img_counter.p", "r"))
replicaCounter = pickle.load(open("replica_counter.p", "r"))

NAME_BASE = "input_images/redgreen_" # This MUST NOT start with IMG

os.system("mv output_images/IMG*.JPG output_images/photo_" + str(imageCounter-1) + "_" + str(replicaCounter) + ".JPG")

os.system("mv output_images/IMG*.CR2 output_images/photo_" + str(imageCounter-1) + "_" + str(replicaCounter) + ".CR2")
    
pickle.dump(replicaCounter+1, open("replica_counter.p", "w"))

