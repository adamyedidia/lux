import os
from best_matrix import padIntegerWithZeros

for i in range(1, 51):
    os.system("ffmpeg -i toeplitz_video_trash/trueocc" + padIntegerWithZeros(i, 2) + \
        ".png -i toeplitz_video_trash/inversionocc" + padIntegerWithZeros(i, 2) + \
        ".png -i toeplitz_video_trash/recovery" + padIntegerWithZeros(i, 2) + \
        ".png -filter_complex hstack toeplitz_video_trash/output" + \
        padIntegerWithZeros(i, 2) + ".png")
