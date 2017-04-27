import rawpy
import imageio

raw = rawpy.imread('right_angle_cam_figs/IMG_3063.CR2')
rgb = raw.postprocess()
imageio.imsave('default.eps', rgb)
