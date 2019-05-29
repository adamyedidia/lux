import cv2
import numpy as np
from utils_prafull import *
import pickle
import glob
import os
import json
import re

if True:
    frames_dict = json.load(open("frames2.txt", "r"))
    print (frames_dict)
    data_dir = "/Users/adamyedidia/dots_vid/"
    folder_name = "joint_folder/"
    dataset = "first_take_synth"
    skip_frames = 4
    seq_name = "movingdisc"

if False:
    frames_dict = json.load(open("frames.txt", "r"))
    data_dir = "/Users/adamyedidia/seagate/"
    folder_name = "first_take/"
    dataset = "first_take"
    skip_frames = 0
    seq_name = "hands"    

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def get_filenames(path, ext):

    print path + "*"

    files = natural_sort(glob.glob(path + "*"))
    print(path)

    print "files", files

    files = [f for f in files if f.endswith(ext)]
    print(len(files))
    return files

# import matplotlib.pyplot as plt

# path: file name beginning such as "/path/to/blah/frame_test_"
# file_start: int which should be the first file read in the sequencce
# file_end: int which should be the last file read in the sequencce
# num_downsample: int which defines number of times the images should be downsampled by half.
def load_pgm_color(path, file_start, file_end, num_downsample):
    images = []
    files = get_filenames(path, ext)

    for i in range(file_start, file_end):
#         print(path + str(i) + ".pgm")
        img = cv2.imread(files[i], cv2.IMREAD_UNCHANGED)

        colored = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2RGB)

        colored = np.array(colored, dtype=np.float32)
        for _ in range(num_downsample):
            colored = downsample_box_half(colored)
        images.append(colored)
    return np.array(images)


def avg_skip_frames(images, skip_frames):
    print("Loaded #images:", len(images))
    N, H, W, C = images.shape
    print(images.shape)
#     Make number of frames multiple of skip_frames
    N -= N % skip_frames
    images = images[0:N]
    print("After chopping: ",  images.shape)
    images = images.reshape(int(len(images)/skip_frames), skip_frames, H, W, C)
    images = np.squeeze(np.mean(images, axis=1))
    
#     Make number of frames a multiple of 8
    N, H, W, C = images.shape
    N -= N % 8
    images = images[0:N]
    print(images.shape)
    return images

def load_pgm_mono(path, file_start, file_end, num_downsample, ext=".pgm"):
    images = []
    files = get_filenames(path, ext)
    overexp_mask = None
    for i in range(file_start, file_end):
#         print(path + str(i) + ext)
        #img = cv2.imread(path + str(i) + ext, cv2.IMREAD_UNCHANGED)
        img = cv2.imread(files[i], cv2.IMREAD_UNCHANGED)
        img = img[0:img.shape[1]-img.shape[1]%(2**num_downsample), 0:img.shape[1]-img.shape[1]%(2**num_downsample)]
        overexp = (img > (2**16 - 100)).astype(np.float32)

        if np.max(np.array(img)) > 2**16 - 1000:
            print('WARNING: frame %i, overexposed pixels %i' % (i, np.sum(img > 2**16-1000)))

        img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2RGB)     # TODO: check that this doesn't do anything super stupid and nonlinear
        colored = np.array(img, dtype=np.float32)
#         print(colored.shape)
        #colored = colored[0:1024,0:1536]
        #img[0:2,0:32] = img[2:4,0:32]   # the annoying blinking thing
        #colored = downsample_box_half_mono(colored) # just get rid of the bayer array by averaging every 2x2 block XXX



        for _ in range(num_downsample):
            colored = downsample_box_half_mono(colored)
            overexp = downsample_box_half_mono(overexp)
        images.append(colored)
        if overexp_mask is not None:
            overexp_mask += overexp
        else:
            overexp_mask = overexp

    overexp_mask = overexp_mask > 0
    
    return np.array(images), overexp_mask

def load_pgm_mono0(path, file_start, file_end, num_downsample, ext=".pgm"):
    images = []
    files = get_filenames(path, ext)
    print(files)
    for i in range(file_start, file_end):
#         print(path + str(i) + ext)
        img = cv2.imread(files[i], cv2.IMREAD_UNCHANGED)
        #img = cv2.imread('%s%04d%s' % (path, i, ext), cv2.IMREAD_UNCHANGED)
        colored = np.array(img, dtype=np.float32)
        #print(colored.shape)
        #colored = colored[0:1024,0:1536]
        #img[0:2,0:32] = img[2:4,0:32]   # the annoying blinking thing
        #colored = downsample_box_half_mono(colored) # just get rid of the bayer array by averaging every 2x2 block XXX

        for _ in range(num_downsample):
            colored = downsample_box_half_mono(colored)
        images.append(colored)
        
    
    images = np.array(images)
    print(images)
    return images

def load_gt(path, file_start, file_end, ext=""):
    images = []
    files = get_filenames(path, '')
    print("Length of gt: ", len(files))
    for i in range(file_start, file_end):
        img = cv2.imread(files[i], cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_NEAREST)
        img = (img - np.min(img))/(np.max(img) - np.min(img)) * 255 
        images.append(img)
    images = np.array(images)
    print("Shape of images: ", images.shape)
    return images

def load_basis():

    imgs = load_pgm_mono(data_dir + filepath + dataset + "_reflector", basis_start, basis_end, n_downsamples)  # 30 rows of delta (two last broken)

    #imgs = load_pgm_mono("E:/passive/data/last_final/lastfinal_reflector_", 3191, 4137, 2)  # hands
    #imgs = load_pgm_mono("E:/passive/data/last_final/lastfinal_reflector_", 3191, 3500, 1)  # hands
    #imgs = load_pgm_mono("E:/passive/data/last_final/lastfinal_reflector_", 3821, 4138, 1)  # spandan hands
    #imgs = load_pgm_mono("E:/passive/data/last_final/lastfinal_reflector_", 4139, 4749, 2)  # spandan walk

    imgs = imgs / 2**8
    print("Print from load_basis: \n", imgs.shape)
    imgs = imgs -np.min((imgs.flatten()))   # XXX
    print(np.min((imgs.flatten())))
    print(np.max((imgs.flatten())))
    print(np.mean((imgs.flatten())))
    print(np.std((imgs.flatten())))

    Z = np.reshape(imgs, [30, 32, imgs.shape[1], imgs.shape[2]])
    Zs = Z.shape
    #Sfull = [1] + Z.shape

    # downsample to half (to save stuff)
    #Z = np.transpose(Z, axes=[2, 3, 0, 1])
    Z = np.reshape(Z, [Zs[0], Zs[1], Zs[2]*Zs[3]])
    Z = downsample_box_half(Z)
    Z = np.reshape(Z, [Zs[0]//2, Zs[1]//2, Zs[2], Zs[3]])
    Z[:,:,0,0:2] =Z[:,:,1,0:2]  # hack away the annoying blinking pixel at corner
    Zs = Z.shape

    #Z = np.transpose(Z, axes=[1,0,2,3])   # h w H W
    if 1:

#         plt.clf()
#         plt.subplot(2, 2, 1)
#         plt.imshow(imgs[0,3:-1,:])
#         plt.subplot(2, 2, 2)
#         plt.imshow(np.transpose(np.squeeze(imgs[:,40,:])))
#         plt.subplot(2, 2, 4)
#         plt.imshow(np.transpose(np.abs(np.squeeze(imgs[1:-1,40,:] - imgs[0:-2,40,:]))))

        #plt.imshow(np.squeeze(Z[:,:,70,150]))

        if 0:
            plt.clf()
            plt.semilogy(S)

        plt.waitforbuttonpress()

        imgs = imgs - np.mean(imgs,0)

#         if 0:/
#             for t in range(imgs.shape[1]):
#                 plt.clf()
                #plt.imshow(np.transpose(np.squeeze(imgs[:, t, :])))
                #plt.imshow(np.tanh(1000*np.transpose(np.squeeze(imgs[:, :, t]))))
                #plt.imshow(np.tanh(1000 * np.transpose(np.squeeze(imgs[:, t,:]))))
#                 plt.imshow(np.tanh(1000 * np.transpose(np.squeeze(imgs[:, t, :]))))
                #plt.imshow(np.tanh(1000 * np.reshape(np.squeeze(imgs[:, t, t]), [30, 32])))
                #plt.imshow(np.transpose(np.abs(np.squeeze(imgs[1:-1, t, :] - imgs[0:-2, t, :]))))
#                 fig.canvas.draw()
#                 fig.canvas.flush_events()
#                 plt.waitforbuttonpress()

#         if 0:
#             for i in range(Z.shape[2]):
#                 for j in range(Z.shape[3]):
#                     plt.clf()
                    #plt.imshow(np.transpose(np.squeeze(imgs[:, t, :])))
                    #plt.imshow(np.tanh(1000*np.transpose(np.squeeze(imgs[:, :, t]))))
                    #plt.imshow(np.tanh(1000 * np.transpose(np.squeeze(imgs[:, t,:]))))
                    #plt.imshow(np.tanh(1000 * np.transpose(np.squeeze(Z[i,j,:, :]))))
#                     plt.imshow(10*((np.squeeze(Z[:, :, i, j]))))
                    #plt.imshow(np.tanh(1000 * np.reshape(np.squeeze(imgs[:, t, t]), [30, 32])))
                    #plt.imshow(np.transpose(np.abs(np.squeeze(imgs[1:-1, t, :] - imgs[0:-2, t, :]))))
#                     fig.canvas.draw()
#                     fig.canvas.flush_events()
                    #plt.waitforbuttonpress()


#         plt.waitforbuttonpress()
        return Z

def load_video(n_downsamples=1):

    #imgs = load_pgm_mono("E:/passive/data/last_final/lastfinal_reflector_", 100, 1123-2*32+1, 2)  # 30 rows of delta (two last broken)

    ####imgs = load_pgm_mono("E:/passive/data/last_final/lastfinal_target_", 3191, 4137, 0)  # hands
    #imgs = load_pgm_mono("E:/passive/data/last_final/lastfinal_target_", 3191, 3820, 0)  # hands
    #imgs = load_pgm_mono("E:/passive/data/last_final/lastfinal_target_", 3191, 3815, 0)  # hands div8
    #imgs = load_pgm_mono("E:/passive/data/last_final/lastfinal_target_", 3821, 4138, 0)  # spandan hands
    #imgs = load_pgm_mono("E:/passive/data/last_final/lastfinal_target_", 4139, 4749, 0)  # spandan walk
    imgs = load_pgm_mono(data_dir + filepath + "/" + dataset + "_target", frames_dict[seq_name][0], frames_dict[seq_name][1], n_downsample)  # spandan walk div8
  # spandan walk div8

    imgs = imgs / 2**8
    print(imgs.shape)
    print(np.max(imgs.flatten()))

    Z = imgs
    Zs = Z.shape

#     plt.clf()
#     plt.imshow(np.transpose(np.squeeze(Z[:, 5, :])))
#     plt.waitforbuttonpress()  # XXX annoying but needed for redraw to work

    if 0:
        for f in range(Zs[0]):
#             plt.clf()
#             plt.imshow(np.squeeze(Z[f,:,:]))
            fig.canvas.draw()
            fig.canvas.flush_events()

    return Z

def skip_frames_gt(images, skip_frames):
    print("Loaded #images:", len(images))
    N, H, W, C = images.shape
    print(images.shape)
#     Make number of frames multiple of skip_frames
    N -= N % skip_frames
    images = images[0:N]
    print("After chopping: ",  images.shape)
    images = images.reshape(int(len(images)/skip_frames), skip_frames, H, W, C)
    images = np.squeeze(images[:, 0, :, :, :])
    
#     Make number of frames a multiple of 8
    N, H, W, C = images.shape
    N -= N % 8
    images = images[0:N]
    print("GTS lenght after chopping and mod 8", images.shape)
    return images
    
def load_observations(n_downsample=3, zero_overexp=True):
    print("skip_frames", skip_frames)
    blacks = None
    blocks_oe = None
    
    if 'black' in frames_dict: 
        blacks, blacks_oe = load_pgm_mono(data_dir + folder_name + dataset + "_reflector", frames_dict['black'][0], frames_dict['black'][1], n_downsample)

    imgs, imgs_oe = load_pgm_mono(data_dir + folder_name + dataset + "_reflector",
                         frames_dict[seq_name][0], frames_dict[seq_name][1], n_downsample)  # lego
    
    gt_imgs = load_gt(data_dir+folder_name+dataset+"_target", frames_dict[seq_name][0], frames_dict[seq_name][1], ext="")
    if skip_frames != 0:
        imgs = avg_skip_frames(imgs, skip_frames)
        gt_imgs = skip_frames_gt(gt_imgs, skip_frames)

    print('max %i' % np.max(imgs))
    print('min %i' % np.min(imgs))

    overexp = imgs > (2**16-50)
    overexp = np.any(overexp, axis=0)


    imgs = imgs / 2**14
    if blacks is not None:
        blacks = blacks / 2**14
        blacks = np.mean(blacks, axis=0, keepdims=True)
        imgs = imgs - blacks
    imgs[:,0,0:32] = imgs[:,1,0:32]

    
    if zero_overexp:
        imgs = imgs * (1.0-np.expand_dims(np.expand_dims(imgs_oe,2),0).astype(np.float32))



    print(imgs.shape)

    print(np.min((imgs.flatten())))
    print(np.max((imgs.flatten())))
    print(np.mean((imgs.flatten())))
    print(np.std((imgs.flatten())))

    Z = imgs
    Zs = Z.shape
    #Sfull = [1] + Z.shape

    # downsample to half (to save stuff)
    #Z = np.transpose(Z, axes=[2, 3, 0, 1])
    Z[:,0,0:2] =Z[:,1,0:2]  # hack away the annoying blinking pixel at corner
    Zs = Z.shape

    #Z = np.transpose(Z, axes=[1,0,2,3])   # h w H W
    if 1:

#         plt.clf()
#         plt.imshow(overexp)
        if 0:
            plt.subplot(2, 2, 1)
            plt.imshow(imgs[0,3:-1,:])
            plt.subplot(2, 2, 2)
            plt.imshow(np.transpose(np.squeeze(imgs[:,40,:])))
            plt.subplot(2, 2, 3)
            plt.subplot(2, 2, 4)
            plt.imshow(np.transpose(np.abs(np.squeeze(imgs[1:-1,40,:] - imgs[0:-2,40,:]))))

            #plt.imshow(np.squeeze(Z[:,:,70,150]))

#         if 0:
#             plt.clf()
#             plt.semilogy(S)

        #plt.waitforbuttonpress()

        #imgs = imgs - np.mean(imgs,0)

#         if 0:
#             for t in range(imgs.shape[1]):
#                 plt.clf()
                #plt.imshow(np.transpose(np.squeeze(imgs[:, t, :])))
                #plt.imshow(np.tanh(1000*np.transpose(np.squeeze(imgs[:, :, t]))))
                #plt.imshow(np.tanh(1000 * np.transpose(np.squeeze(imgs[:, t,:]))))
#                 plt.imshow(np.tanh(1000 * np.transpose(np.squeeze(imgs[:, t, :]))))
                #plt.imshow(np.tanh(1000 * np.reshape(np.squeeze(imgs[:, t, t]), [30, 32])))
                #plt.imshow(np.transpose(np.abs(np.squeeze(imgs[1:-1, t, :] - imgs[0:-2, t, :]))))
#                 fig.canvas.draw()
#                 fig.canvas.flush_events()
#                 plt.waitforbuttonpress()

#         if 0:
#             for i in range(Z.shape[2]):
#                 for j in range(Z.shape[3]):
#                     plt.clf()
                    #plt.imshow(np.transpose(np.squeeze(imgs[:, t, :])))
                    #plt.imshow(np.tanh(1000*np.transpose(np.squeeze(imgs[:, :, t]))))
                    #plt.imshow(np.tanh(1000 * np.transpose(np.squeeze(imgs[:, t,:]))))
                    #plt.imshow(np.tanh(1000 * np.transpose(np.squeeze(Z[i,j,:, :]))))
#                     plt.imshow(10*((np.squeeze(Z[:, :, i, j]))))
                    #plt.imshow(np.tanh(1000 * np.reshape(np.squeeze(imgs[:, t, t]), [30, 32])))
                    #plt.imshow(np.transpose(np.abs(np.squeeze(imgs[1:-1, t, :] - imgs[0:-2, t, :]))))
#                     fig.canvas.draw()
#                     fig.canvas.flush_events()
                    #plt.waitforbuttonpress()


#         plt.waitforbuttonpress()
        return Z, imgs_oe, gt_imgs    # [frame, H, W]


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
# print(device)

print("1")

obs = load_observations()

print("2")

pickle.dump(obs, open("prafull_obs.p", "w"))
