import numpy as np
from video_magnifier import viewFrame, viewFrameR, viewFlatFrame
import pickle
import matplotlib.pyplot as p
from best_matrix import padIntegerWithZeros

#video1 = pickle.load(open("darpa_vid_gt.p", "r"))
#video2 = pickle.load(open("darpa_simple_recon.p", "r"))
#video3 = pickle.load(open("darpa_vert_recon.p", "r"))
#video4 = pickle.load(open("darpa_combined_recon.p", "r"))

video1 = pickle.load(open("darpa_vid_gt.p", "r"))
video2 = pickle.load(open("darpa_fan_recon_noisy.p", "r"))
video3 = pickle.load(open("darpa_vert_recon_noisy.p", "r"))
video4 = pickle.load(open("darpa_combined_recon_noisy.p", "r"))

print(len(video1), len(video2))

video1Start = 11.93
video1Rate = 1.0

video2Start = 0
video2Rate = 2.0

video3Start = 0
video3Rate = 2.0

video4Start = 0
video4Rate = 2.0

video1Counter = video1Start
video2Counter = video2Start
video3Counter = video3Start
video4Counter = video4Start
overallCounter = 0

while video1Counter < len(video1) - 1 and video2Counter < len(video2) - 1:
	frame1 = video1[int(video1Counter)]
	frame2 = video2[int(video2Counter)]
	frame3 = video3[int(video3Counter)]
	frame4 = video4[int(video4Counter)]

	print(overallCounter)

	p.clf()

	p.subplot(221)
	p.axis("off")
	viewFrame(frame1, filename="pass", relax=True, differenceImage=False, magnification=1)

	p.subplot(222)
	p.axis("off")
	viewFrame(np.flip(frame2, 0), filename="pass", adaptiveScaling=True,
		relax=True, magnification=1, differenceImage=False)

	p.subplot(223)
	p.axis("off")
	viewFrame(np.flip(frame3, 0), filename="pass", adaptiveScaling=True,
		relax=True, magnification=1, differenceImage=False)

	p.subplot(224)
	p.axis("off")
	viewFrame(np.flip(frame4, 0), filename="pass", adaptiveScaling=True,
		relax=True, magnification=1, differenceImage=False)

	p.savefig("aligned_vid/frame_" + padIntegerWithZeros(overallCounter, 3) + ".png")

	overallCounter += 1
	video1Counter += video1Rate
	video2Counter += video2Rate
	video3Counter += video3Rate
	video4Counter += video4Rate



