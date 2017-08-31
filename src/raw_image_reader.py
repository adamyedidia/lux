import rawpy
import imageio

raw = rawpy.imread('right_angle_cam_figs/IMG_3056.CR2')
rgb = raw.postprocess()
imageio.imsave('japan_flag_garbled_eps.eps', rgb)
