
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf


import pickle
from math import log, floor

import matplotlib.pyplot as plt

import sys

import numpy as np

GEN = False
MAG = True
ONLINE_TEST = False
ONLINE_TEST_2 = False
WINDOW_IM = False

IM_SIZE = 28
moniker = "dora_windowed"

def hanning2D(shape):
    print (shape)

    assert len(shape) == 2

    return np.dot(np.reshape(np.hanning(shape[0]), (shape[0], 1)), \
        np.reshape(np.hanning(shape[1]), (1, shape[1])))

def windowFrame(frame):
    frameDims = frame.shape[:-1]

#   viewFrame(imageify(hanning2D(windowDims)))

    return np.multiply(imageify(hanning2D(frameDims)), frame)

def windowBWFrame(frame):
    frameDims = frame.shape

#   viewFrame(imageify(hanning2D(windowDims)))

    return np.multiply(hanning2D(frameDims), frame)

def imageify(arr):
    result = np.reshape(np.kron(arr, np.array([255,255,255])), arr.shape + tuple([3]))

    return result

def padIntegerWithZeros(x, maxLength):
    if x == 0:
        return "0"*maxLength

    eps = 1e-8


    assert log(x+0.0001, 10) < maxLength

    return "0"*(maxLength-int(floor(log(x, 10)+eps))-1) + str(x)

def getMags(arr):
    if len(arr.shape) == 3:
        assert arr.shape[2] == 1

        return np.reshape(np.abs(np.fft.fft2(arr[:,:,0])), arr.shape)
    else:
        return np.abs(np.fft.fft2(arr))

def getMagsK(arr):
#    print(type(arr))

#    if type(arr) == type(np.array([1,1])):

#        plt.matshow(tf.spectral.rfft2d(arr))
#        plt.show()

    return K.abs(tf.spectral.rfft2d(arr))

def oppositeLoss(y_true,y_pred):
    return K.mean((y_true + y_pred)**2)

def squaredErrorLoss(y_true,y_pred):
    return K.mean((y_true - y_pred)**2)

def absoluteLoss(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))

def magLoss(y_true, y_pred):
    return squaredErrorLoss(y_true, getMagsK(y_pred))
#    return absoluteLoss(y_true, getMagsK(y_pred))

def doubleMagLoss(y_true, y_pred):
 #   print(y_true)
 #   print(y_pred)

    return (getMagsK(y_true) - getMagsK(y_pred))**2


class GAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(100,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        noise_shape = (100,)

        model = Sequential()

        img_shape = (self.img_rows, self.img_cols, self.channels)

        model = Sequential()

        model.add(Flatten(input_shape=img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (half_batch, 100))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("gan/images/mnist_%d.png" % epoch)
        plt.close()

class myGenerator():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the generator
        self.generator = self.build_generator()
#        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)
#        self.generator.compile(loss=squaredErrorLoss, optimizer=optimizer)
#        self.generator.compile(loss=magLoss, optimizer=optimizer)
        self.generator.compile(loss=doubleMagLoss, optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(100,))
        img = self.generator(z)

    def build_generator(self):

        noise_shape = (100,)

        model = Sequential()

        model.add(Dense(256, input_shape=noise_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)

        dora = np.reshape((pickle.load(open("dora_28_28.p", "rb")) - 127.5*np.ones((28, 28)))/127.5, (28, 28, 1))
        doraMags = getMags(dora)

        doraList = np.array([dora]*batch_size)
        doraMagList = np.array([doraMags]*batch_size)

#        print(dora)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
#            idx = np.random.randint(0, X_train.shape[0], half_batch)
#            imgs = X_train[idx]

#            noise = np.random.normal(0, 1, (half_batch, 100))

            # Generate a half batch of new images
#            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
#            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
#            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
#            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.generator.train_on_batch(noise, doraList)
#            g_loss = self.generator.train_on_batch(noise, doraMagList)

            # Plot the progress
            print ("%d [G loss: %f]" % (epoch, g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        direc = "mags/"
        moniker = "dora_windowed"


        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        genPickle = self.generator.predict(np.random.normal(0,1,(1,100)))

        # Rescale images 0 - 1
        genPickle = 0.5 * genPickle + 0.5

        pickle.dump(genPickle, open(direc + moniker + "_" + padIntegerWithZeros(int(epoch/200), 3) + ".p", "wb"))
        
#        print(genPickle.shape)

        fig, axs = plt.subplots(1, 1)
        axs.imshow(genPickle[0,:,:,0], cmap="gray")
        fig.savefig(direc + moniker + "_" + padIntegerWithZeros(int(epoch/200), 3) + ".png")
        plt.close()

#        open("gan/images/dora_%d.png" % int(epoch/200)), "wb")




        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(direc + moniker + "_grid_" + padIntegerWithZeros(int(epoch/200), 3) + ".png")
        plt.close()

class magGenerator():
    def __init__(self):
        self.img_rows = IM_SIZE
        self.img_cols = IM_SIZE
        self.img_dims = (self.img_rows, self.img_cols)
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
#        self.img_shape = (self.img_rows, self.img_cols)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the generator
        self.generator = self.build_generator()
#        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)
        self.generator.compile(loss=squaredErrorLoss, optimizer=optimizer)
#        self.generator.compile(loss=doubleMagLoss, optimizer=optimizer)
#        self.generator.compile(loss=doubleMagLoss, optimizer=optimizer)
#        self.generator.compile(loss=oppositeLoss, optimizer=optimizer)


        # The generator takes noise as input and generated imgs
        z = Input(shape=(100,))
        img = self.generator(z)

    def build_generator(self):

        noise_shape = (100,)

        model = Sequential()

#        inputLayer = Dense(np.prod(self.img_shape), input_shape=noise_shape)
        inputLayer = Dense(256, input_shape=noise_shape)
        inputLayer.name = "inp_layer"

        imgOutLayer = Reshape(self.img_dims)
        imgOutLayer.name = "img_out"

#        fourierLayer = Lambda(tf.spectral.rfft2d)
 #       fourierLayer.name = "fft_layer"

#        fourierLayer = Lambda(tf.spectral.fft2d, dtype='complex64')
#        fourierLayer = Lambda(lambda x: tf.spectral.fft2d(tf.dtypes.complex(x, \
 #           tf.zeros(self.img_shape))))
        fourierLayer = Lambda(lambda x: tf.signal.fft2d(tf.cast(x, \
            tf.complex64)))
#        fourierLayer = Lambda(lambda x: np.fft.fft2(tf.cast(x, \
#            tf.complex64)))
        fourierLayer.name = "fft_layer"

        absLayer = Lambda(K.abs)
        absLayer.name = "abs_layer"

        model.add(inputLayer)
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(imgOutLayer)
#        model.add(Lambda(lambda x: tf.as_complex))
#        model.add(Lambda(lambda x: tf.dtypes.complex(x, tf.zeros(self.img_shape))))
        model.add(fourierLayer)
        model.add(absLayer)
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def train(self, epochs, batch_size=128, save_interval=50):
        # Load the dataset
#        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
#        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
#        X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)

        doraZeroOne = np.reshape((pickle.load(open(moniker + "_" + str(IM_SIZE) + "_" + str(IM_SIZE) + ".p", "rb")))/255, \
            (IM_SIZE, IM_SIZE, 1))
        dora = pickle.load(open(moniker + "_" + str(IM_SIZE) + "_" + str(IM_SIZE) + ".p", "rb")) 

        doraAvg = np.sum(dora)/(IM_SIZE*IM_SIZE)
        print(doraAvg)
        dora = np.reshape((dora - \
            doraAvg*np.ones((IM_SIZE, IM_SIZE)))/doraAvg, (IM_SIZE, IM_SIZE, 1))

        doraMags = getMags(dora)

        fig, axs = plt.subplots(1, 1)
        axs.imshow(doraMags[:,:,0], cmap="gray")
        fig.savefig("first.png")
        plt.close()

        doraZeroOneList = np.array([doraZeroOne]*batch_size)
        doraList = np.array([dora]*batch_size)
        doraMagList = np.array([doraMags]*batch_size)

        fig, axs = plt.subplots(1, 1)
        axs.imshow(doraMagList[0,:,:,0], cmap="gray")
        fig.savefig("second.png")
        plt.close()

#        print(dora)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
#            idx = np.random.randint(0, X_train.shape[0], half_batch)
#            imgs = X_train[idx]

#            noise = np.random.normal(0, 1, (half_batch, 100))

            # Generate a half batch of new images
#            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
#            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
#            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
#            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
#            g_loss = self.generator.train_on_batch(noise, doraZeroOneList)
#            g_loss = self.generator.train_on_batch(noise, doraList)
            g_loss = self.generator.train_on_batch(noise, doraMagList)

            # Plot the progress
            print ("%d [G loss: %f]" % (epoch, g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        direc = "mags/"

        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        littleNoise = np.random.normal(0,1,(1,100))
#        littleNoise = np.zeros((1,100))

        genPickle = self.generator.predict(littleNoise)

        pickle.dump(genPickle, open(direc + moniker + "_" + padIntegerWithZeros(int(epoch/200), 3) + ".p", "wb"))

        dora = pickle.load(open(moniker + "_" + str(IM_SIZE) + "_" + str(IM_SIZE) + ".p", "rb")) 

        doraAvg = np.sum(dora)/(IM_SIZE*IM_SIZE)
        print(doraAvg)
        dora = np.reshape((dora - \
            doraAvg*np.ones((IM_SIZE, IM_SIZE)))/doraAvg, (IM_SIZE, IM_SIZE, 1))
        doraMags = getMags(dora)

#        print(doraMags.shape)

        fig, axs = plt.subplots(1, 1)
        axs.imshow(doraMags[:,:,0], cmap="gray")
        fig.savefig(moniker + "_mags.png")
        plt.close()        

#        print(doraMags[:,:,0])
#        print(genPickle[0,:,:,0])

        # Rescale images 0 - 1
#        genPickle = 0.5 * genPickle + 0.5
        
#        print(genPickle.shape)

        fig, axs = plt.subplots(1, 1)
        axs.imshow(genPickle[0,:,:,0], cmap="gray")
        fig.savefig(direc + moniker + "_" + padIntegerWithZeros(int(epoch/200), 3) + ".png")
        plt.close()

        pickle.dump(genPickle[0,:,:,0], open(direc + moniker + "_" + padIntegerWithZeros(int(epoch/200), 3) + ".p", "wb"))

#        open("gan/images/dora_%d.png" % int(epoch/200)), "wb")

#        print([layer.name for layer in self.generator.layers[1].layers])

        intermediate_layer_model = Model(inputs=self.generator.layers[1].get_layer("inp_layer").input,
                                         outputs=self.generator.layers[1].get_layer("img_out").output)
        intermediate_output = intermediate_layer_model.predict(littleNoise)
#        print(intermediate_output[0,:,:])

#        print(0)
#        plt.matshow(intermediate_output[0,:,:])
#        plt.show()

        fourierLayerModel = Model(inputs=self.generator.layers[1].get_layer("inp_layer").input,
            outputs=self.generator.layers[1].get_layer("fft_layer").output)
        fourierOutput = fourierLayerModel.predict(littleNoise)

#        print(fourierOutput.shape)
#        print(intermediate_output.shape)

#        print(fourierOutput.shape)
#        print(intermediate_output.shape)

#        print("a")
#        plt.matshow(np.abs(fourierOutput[0,:,:]))
#        plt.show()  

        absLayerModel = Model(inputs=self.generator.layers[1].get_layer("inp_layer").input,
            outputs=self.generator.layers[1].get_layer("abs_layer").output)
        absOutput = absLayerModel.predict(littleNoise)

#        print("a1")                  

#        plt.matshow(absOutput[0,:,:])
#        plt.show()          

#        print("b")
#        plt.matshow(np.abs(np.fft.fft2(intermediate_output[0,:,:])))
#        plt.show()

#        sess = tf.Session()

#        tf_ft_testimage = tf.spectral.fft2d(intermediate_output[0,:,:])
#        tf_ft_testimage = tf.spectral.fft2d(tf.cast(intermediate_output[0,:,:,0], \
#            tf.complex64))
#        print(tf_ft_testimage.eval(session=sess))

#        for i in range(6):
#            print(i)
#        plt.matshow(np.abs(tf_ft_testimage.eval(session=sess)))
#        plt.show()                

#        print("fourierOutput")
 #       print(fourierOutput[0,:,:])

#        print("computed")
#        print(tf_ft_testimage.eval(session=sess))


        fig, axs = plt.subplots(1, 1)
        axs.imshow(intermediate_output[0,:,:], cmap="gray")
        fig.savefig(direc + moniker + "_img_" + padIntegerWithZeros(int(epoch/200), 3) + ".png")
        plt.close()

        pickle.dump(intermediate_output[0,:,:], open(direc + moniker + "_img_" + \
            padIntegerWithZeros(int(epoch/200), 3) + ".p", "wb"))

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(direc + moniker + "_grid_" + padIntegerWithZeros(int(epoch/200), 3) + ".png")
        plt.close()

class convGenerator():
    def __init__(self):
        self.img_rows = IM_SIZE
        self.img_cols = IM_SIZE
        self.img_dims = (self.img_rows, self.img_cols)
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
#        self.img_shape = (self.img_rows, self.img_cols)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the generator
        self.generator = self.build_generator()
#        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)
        self.generator.compile(loss=squaredErrorLoss, optimizer=optimizer)
#        self.generator.compile(loss=doubleMagLoss, optimizer=optimizer)
#        self.generator.compile(loss=doubleMagLoss, optimizer=optimizer)
#        self.generator.compile(loss=oppositeLoss, optimizer=optimizer)


        # The generator takes noise as input and generated imgs
        z = Input(shape=(100,))
        img = self.generator(z)

    def build_generator(self):

        noise_shape = (100,)

        model = Sequential()

#        inputLayer = Dense(np.prod(self.img_shape), input_shape=noise_shape)
        inputLayer = Dense(256, input_shape=noise_shape)
        inputLayer.name = "inp_layer"

        imgOutLayer = Reshape(self.img_dims)
        imgOutLayer.name = "img_out"

#        fourierLayer = Lambda(tf.spectral.rfft2d)
 #       fourierLayer.name = "fft_layer"

#        fourierLayer = Lambda(tf.spectral.fft2d, dtype='complex64')
#        fourierLayer = Lambda(lambda x: tf.spectral.fft2d(tf.dtypes.complex(x, \
 #           tf.zeros(self.img_shape))))
        fourierLayer = Lambda(lambda x: tf.signal.fft2d(tf.cast(x, \
            tf.complex64)))
#        fourierLayer = Lambda(lambda x: np.fft.fft2(tf.cast(x, \
#            tf.complex64)))
        fourierLayer.name = "conv_layer"

        absLayer = Lambda(K.abs)
        absLayer.name = "abs_layer"

        model.add(inputLayer)
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape*2), activation='tanh'))
        model.add(imgOutLayer)
#        model.add(Lambda(lambda x: tf.as_complex))
#        model.add(Lambda(lambda x: tf.dtypes.complex(x, tf.zeros(self.img_shape))))
        model.add(fourierLayer)
        model.add(absLayer)
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def train(self, epochs, batch_size=128, save_interval=50):
        # Load the dataset
#        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
#        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
#        X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)

        doraZeroOne = np.reshape((pickle.load(open(moniker + "_" + str(IM_SIZE) + "_" + str(IM_SIZE) + ".p", "rb")))/255, \
            (IM_SIZE, IM_SIZE, 1))
        dora = pickle.load(open(moniker + "_" + str(IM_SIZE) + "_" + str(IM_SIZE) + ".p", "rb")) 

        doraAvg = np.sum(dora)/(IM_SIZE*IM_SIZE)
        print(doraAvg)
        dora = np.reshape((dora - \
            doraAvg*np.ones((IM_SIZE, IM_SIZE)))/doraAvg, (IM_SIZE, IM_SIZE, 1))

        doraMags = getMags(dora)

        fig, axs = plt.subplots(1, 1)
        axs.imshow(doraMags[:,:,0], cmap="gray")
        fig.savefig("first.png")
        plt.close()

        doraZeroOneList = np.array([doraZeroOne]*batch_size)
        doraList = np.array([dora]*batch_size)
        doraMagList = np.array([doraMags]*batch_size)

        fig, axs = plt.subplots(1, 1)
        axs.imshow(doraMagList[0,:,:,0], cmap="gray")
        fig.savefig("second.png")
        plt.close()

#        print(dora)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
#            idx = np.random.randint(0, X_train.shape[0], half_batch)
#            imgs = X_train[idx]

#            noise = np.random.normal(0, 1, (half_batch, 100))

            # Generate a half batch of new images
#            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
#            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
#            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
#            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
#            g_loss = self.generator.train_on_batch(noise, doraZeroOneList)
#            g_loss = self.generator.train_on_batch(noise, doraList)
            g_loss = self.generator.train_on_batch(noise, doraMagList)

            # Plot the progress
            print ("%d [G loss: %f]" % (epoch, g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        direc = "mags/"

        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        littleNoise = np.random.normal(0,1,(1,100))
#        littleNoise = np.zeros((1,100))

        genPickle = self.generator.predict(littleNoise)

        pickle.dump(genPickle, open(direc + moniker + "_" + padIntegerWithZeros(int(epoch/200), 3) + ".p", "wb"))

        dora = pickle.load(open(moniker + "_" + str(IM_SIZE) + "_" + str(IM_SIZE) + ".p", "rb")) 

        doraAvg = np.sum(dora)/(IM_SIZE*IM_SIZE)
        print(doraAvg)
        dora = np.reshape((dora - \
            doraAvg*np.ones((IM_SIZE, IM_SIZE)))/doraAvg, (IM_SIZE, IM_SIZE, 1))
        doraMags = getMags(dora)

#        print(doraMags.shape)

        fig, axs = plt.subplots(1, 1)
        axs.imshow(doraMags[:,:,0], cmap="gray")
        fig.savefig(moniker + "_mags.png")
        plt.close()        

#        print(doraMags[:,:,0])
#        print(genPickle[0,:,:,0])

        # Rescale images 0 - 1
#        genPickle = 0.5 * genPickle + 0.5
        
#        print(genPickle.shape)

        fig, axs = plt.subplots(1, 1)
        axs.imshow(genPickle[0,:,:,0], cmap="gray")
        fig.savefig(direc + moniker + "_" + padIntegerWithZeros(int(epoch/200), 3) + ".png")
        plt.close()

        pickle.dump(genPickle[0,:,:,0], open(direc + moniker + "_" + padIntegerWithZeros(int(epoch/200), 3) + ".p", "wb"))

#        open("gan/images/dora_%d.png" % int(epoch/200)), "wb")

#        print([layer.name for layer in self.generator.layers[1].layers])

        intermediate_layer_model = Model(inputs=self.generator.layers[1].get_layer("inp_layer").input,
                                         outputs=self.generator.layers[1].get_layer("img_out").output)
        intermediate_output = intermediate_layer_model.predict(littleNoise)
#        print(intermediate_output[0,:,:])

#        print(0)
#        plt.matshow(intermediate_output[0,:,:])
#        plt.show()

        fourierLayerModel = Model(inputs=self.generator.layers[1].get_layer("inp_layer").input,
            outputs=self.generator.layers[1].get_layer("fft_layer").output)
        fourierOutput = fourierLayerModel.predict(littleNoise)

#        print(fourierOutput.shape)
#        print(intermediate_output.shape)

#        print(fourierOutput.shape)
#        print(intermediate_output.shape)

#        print("a")
#        plt.matshow(np.abs(fourierOutput[0,:,:]))
#        plt.show()  

        absLayerModel = Model(inputs=self.generator.layers[1].get_layer("inp_layer").input,
            outputs=self.generator.layers[1].get_layer("abs_layer").output)
        absOutput = absLayerModel.predict(littleNoise)

#        print("a1")                  

#        plt.matshow(absOutput[0,:,:])
#        plt.show()          

#        print("b")
#        plt.matshow(np.abs(np.fft.fft2(intermediate_output[0,:,:])))
#        plt.show()

#        sess = tf.Session()

#        tf_ft_testimage = tf.spectral.fft2d(intermediate_output[0,:,:])
#        tf_ft_testimage = tf.spectral.fft2d(tf.cast(intermediate_output[0,:,:,0], \
#            tf.complex64))
#        print(tf_ft_testimage.eval(session=sess))

#        for i in range(6):
#            print(i)
#        plt.matshow(np.abs(tf_ft_testimage.eval(session=sess)))
#        plt.show()                

#        print("fourierOutput")
 #       print(fourierOutput[0,:,:])

#        print("computed")
#        print(tf_ft_testimage.eval(session=sess))


        fig, axs = plt.subplots(1, 1)
        axs.imshow(intermediate_output[0,:,:], cmap="gray")
        fig.savefig(direc + moniker + "_img_" + padIntegerWithZeros(int(epoch/200), 3) + ".png")
        plt.close()

        pickle.dump(intermediate_output[0,:,:], open(direc + moniker + "_img_" + \
            padIntegerWithZeros(int(epoch/200), 3) + ".p", "wb"))

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(direc + moniker + "_grid_" + padIntegerWithZeros(int(epoch/200), 3) + ".png")
        plt.close()


if __name__ == '__main__':
    if GEN:
        gen = myGenerator()
        gen.train(epochs=30000, batch_size=32, save_interval=200)

    if MAG:
        gen = magGenerator()
        gen.train(epochs=30000, batch_size=32, save_interval=200)

    if ONLINE_TEST:
        inp = Input(shape=(299,299),dtype='complex64')
        tensorTransformada = Lambda(tf.fft2d, output_shape=(None, 299, 299))(inp)

    if WINDOW_IM:
        imName = "dora_28_28"

        im = pickle.load(open(imName + ".p", "rb"))

        print (im.shape)

        windowedIm = windowBWFrame(im)

        pickle.dump(np.array(windowedIm), open(imName + "_windowed.p", "wb"))

    if ONLINE_TEST_2:
        # check if np.fft2d of TF.fft2d and NP have the same result

        testimage = np.random.rand(6, 6)

        print (testimage)

        sess = tf.Session()

        ft_testimage = np.fft.fft2(testimage)
        plt.matshow(np.abs(ft_testimage))
        plt.show()
#        np_result = np.sum(ft_testimage)

        tf_ft_testimage = tf.fft2d(testimage)
        plt.matshow(np.abs(tf_ft_testimage.eval(session=sess)))
        plt.show()        
#        tf_result = np.sum(tf_ft_testimage.eval(session=sess))


