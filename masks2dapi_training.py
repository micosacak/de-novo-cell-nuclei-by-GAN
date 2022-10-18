import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from IPython import display
import copy as cc
import re as rr
import random
import tensorflow as tf
import scipy
import datetime
import inspect

# please not that some codes are directly used or modified from Tensorflow website 
# https://www.tensorflow.org/tutorials/generative/pix2pix (last accessed:1st October, 2022)

LAMBDA = 1
# some functions for image processing
def getByC(the_C_image):
	if the_C_image.shape[len(the_C_image.shape)-1] == 1:
		return the_C_image[...,0]
	else:
		brk = np.ones((the_C_image.shape[0], 5))*-2
		the_conc_C = np.concatenate((brk,the_C_image[...,0], brk, the_C_image[...,1]), axis = 1)
		if the_C_image.shape[2] >= 3:
			for i in range(2, the_C_image.shape[2]):
				the_conc_C = np.concatenate((the_conc_C, brk,the_C_image[...,i]), axis = 1)	
		the_conc_C = np.concatenate((the_conc_C, brk), axis = 1)
		return the_conc_C

def add_breaks(the_image):
	if len(the_image.shape) == 3:
		t0 = getByC(the_image)
	elif len(the_image.shape) == 4:
		t0 = getByC(the_image[0,...])
		brk = np.ones((5,t0.shape[1]))*-2
		if the_image.shape[0] >= 2:
			for i in range(1,the_image.shape[0]):
				t1 = getByC(the_image[i,...])
				t0 = np.concatenate((t0,brk,t1), axis = 0)
	t0[t0 == -2] = 255
	return t0

def pp(x):
	plt.imshow(x)
	plt.show()


# set GPU or CPU
def setGPU(idx = 0):
	gpus = tf.config.list_physical_devices('GPU')
	if gpus:
  		# Restrict TensorFlow to only use the first GPU
  		try:
    			tf.config.set_visible_devices(gpus[idx], 'GPU')
    			logical_gpus = tf.config.list_logical_devices('GPU')
    			print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  		except RuntimeError as e:
    			# Visible devices must be set before GPUs have been initialized
    			print(e)

def setCPU(num_threads = 8):
	os.environ["OMP_NUM_THREADS"] = str(num_threads)
	os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
	os.environ["TF_NUM_INTEROP_THREADS"] = str(num_threads)
	tf.config.threading.set_inter_op_parallelism_threads(num_threads)
	tf.config.threading.set_intra_op_parallelism_threads(num_threads)
	tf.config.set_soft_device_placement(True)

# some functions for plotting.
def rm_str(the_string):
	if isinstance(the_string, list):
		if len(the_string) == 0:
			return 0
		else:
			aa = [rr.sub("[a-zA-Z_.]","", x) for x in the_string]
			aa = [int(x) for x in aa]
			return max(aa) + 1
	elif isinstance(the_string, str):
		return int(rr.sub("[a-zA-Z]_.","", the_string))
	else:
		raise ValueError("Unknown type: {}, the type must be a list or a string".format(type(the_string)))

def get_indexes(my_list, prefix = None, suffix = None):
	my_list = [x.replace(prefix,"",) for x in my_list]
	my_list = [int(x.strip(suffix)) for x in my_list]
	return np.argsort(my_list)


def generate_images(model, test_input, tar, epoch):
	the_string = os.listdir(".")
	if "image_at_epoch_99999.png" in the_string:
		the_string.remove("image_at_epoch_99999.png")
	the_string = [x for x in the_string if x.endswith(".png")]
	xx = rm_str(the_string)
	prediction = model(test_input, training = True)
	prediction = prediction.numpy()#[:,:,:,0]
	#prediction[prediction > 0] = 255.0
	prediction = (prediction + 1.0) * 127.5
	prediction  = add_breaks(prediction)
	test_input = cc.copy(test_input)
	test_input = (test_input + 1.0) * 127.5
	test_input = test_input * 2
	test_input[test_input > 255] = 255
	tar = (tar + 1.0)*127.5
	img_left = np.concatenate((test_input, tar), axis = 3)
	img_left = add_breaks(img_left)
	img_final = np.concatenate((img_left, prediction), axis = 1)
	cv2.imwrite('image_at_epoch_{:05d}.png'.format(xx), img_final)
	cv2.imwrite('image_at_epoch_99999.png', img_final)

# checked on 20220616. OK.
def get_crop_shape(target, query):
	# the height
	channelHeight = target.get_shape()[1] - query.get_shape()[1]
	assert (channelHeight >= 0)
	channelHeight1 = int(channelHeight/2)
	if channelHeight % 2 != 0:
		channelHeight2 = channelHeight1 + 1
	else:
		channelHeight2 = channelHeight1
	# the width
	channelWidth = target.get_shape()[2] - query.get_shape()[2]
	assert (channelWidth >= 0)
	channelWidth1 = int(channelWidth/2)
	if channelWidth % 2 != 0:
		channelWidth2 = channelWidth1 + 1
	else:
		channelWidth2 = channelWidth1
	return (channelHeight1, channelHeight2), (channelWidth1, channelWidth2)

# checked on 20220616. OK.
def getAct(x):
	return tf.keras.layers.LeakyReLU()(x)

def getCLB(the_input, filterSize = 16, kernelSize = 2, strd = 2, pdng = "same", useBs = False):
	clb1 = tf.keras.layers.Conv2D(filterSize, kernelSize, strides = strd, padding= pdng, kernel_initializer="he_normal", use_bias=useBs)(the_input)
	clb1 = tf.keras.layers.BatchNormalization()(clb1)
	clb1 = tf.keras.layers.LeakyReLU()(clb1)
	return clb1

def getCLU(the_input, filterSize = 16, kernelSize = 2, strd = 2, pdng = "same", useBs = False, drop_me = False, drop_level = 0.5): 
	clu1 = tf.keras.layers.Conv2DTranspose(filterSize, kernel_size = kernelSize, strides = strd, padding=pdng, kernel_initializer="he_normal",use_bias=useBs)(the_input)
	clu1 = tf.keras.layers.BatchNormalization()(clu1)
	if drop_me:
		clu1 = tf.keras.layers.Dropout(drop_level)(clu1)
	clu1 = tf.keras.layers.LeakyReLU()(clu1)
	return clu1

def getMerged(target_layer, up_layer):
	ch, cw = get_crop_shape(target_layer, up_layer)
	up_layer = tf.keras.layers.ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(up_layer) # add zeropadding.
	merged_layer = tf.keras.layers.concatenate([target_layer,up_layer], axis = 3)
	return merged_layer

def getUnet6(the_input_layer, nChannels = 5, actFunc = "tanh", filterSize = 16, kernelSize = 3, strd = 2, pdng = "same", useBs = False, drpL = 0.5):	
	convL1 = tf.keras.layers.Conv2D(filterSize*1, kernelSize,strides = strd, padding='same', kernel_initializer="he_normal", use_bias=False)(the_input_layer)
	convL1 = tf.keras.layers.LeakyReLU()(convL1)
	convL2 = getCLB(convL1, filterSize = filterSize*2, kernelSize = kernelSize, strd = strd)
	convL3 = getCLB(convL2, filterSize = filterSize*4, kernelSize = kernelSize, strd = strd)
	convL4 = getCLB(convL3, filterSize = filterSize*8, kernelSize = kernelSize, strd = strd)
	convL5 = getCLB(convL4, filterSize = filterSize*16, kernelSize = kernelSize, strd = strd)
	convL6 = getCLB(convL5, filterSize = filterSize*32, kernelSize = kernelSize, strd = strd)
	convL7 = getCLB(convL6, filterSize = filterSize*64, kernelSize = kernelSize, strd = strd)
	# now deconvL layer and concatenation
	deconvL8 = getCLU(convL7,filterSize = filterSize*32, kernelSize = kernelSize, strd = strd, drop_me = True, drop_level = drpL)
	cn8 = tf.keras.layers.Concatenate()([deconvL8, convL6])
	#
	deconvL9 = getCLU(cn8,filterSize = filterSize*16, kernelSize = kernelSize, strd = strd, drop_me = True, drop_level = drpL)
	cn9 = tf.keras.layers.Concatenate()([deconvL9, convL5])
	#
	deconvL10 = getCLU(cn9,filterSize = filterSize*8, kernelSize = kernelSize, strd = strd, drop_me = False, drop_level = drpL)
	cn10 = tf.keras.layers.Concatenate()([deconvL10, convL4])
	#
	deconvL11 = getCLU(cn10,filterSize = filterSize*4, kernelSize = kernelSize, strd = strd, drop_me = False, drop_level = drpL)
	cn11 = tf.keras.layers.Concatenate()([deconvL11, convL3])
	#
	deconvL12 = getCLU(cn11,filterSize = filterSize*2, kernelSize = kernelSize, strd = strd, drop_me = False, drop_level = drpL)
	cn12 = tf.keras.layers.Concatenate()([deconvL12, convL2])
	#
	deconvL13 = getCLU(cn12,filterSize = filterSize*1, kernelSize = kernelSize, strd = strd, drop_me = False, drop_level = drpL)
	cn13 = tf.keras.layers.Concatenate()([deconvL13, convL1])
	#
	outLayer = tf.keras.layers.Conv2DTranspose(nChannels, kernelSize, strides = strd, padding=pdng, kernel_initializer="he_normal", activation=actFunc)(cn13) # use tanh, softsign, selu, elu
	return outLayer

def getUnet4(the_input_layer, nChannels = 5, actFunc = "tanh", filterSize = 16, kernelSize = 3, strd = 2, pdng = "same", useBs = False, drpL = 0.5):	
	convL1 = tf.keras.layers.Conv2D(filterSize, kernelSize,strides = strd, padding='same', kernel_initializer="he_normal", use_bias=False)(the_input_layer)
	convL1 = tf.keras.layers.LeakyReLU()(convL1)
	convL2 = getCLB(convL1, filterSize = filterSize*2, kernelSize = kernelSize, strd = strd)
	convL3 = getCLB(convL2, filterSize = filterSize*4, kernelSize = kernelSize, strd = strd)
	convL4 = getCLB(convL3, filterSize = filterSize*8, kernelSize = kernelSize, strd = strd)
	convL5 = getCLB(convL4, filterSize = filterSize*16, kernelSize = kernelSize, strd = strd)
	# deconv layers
	deconvL6 = getCLU(convL5,filterSize = filterSize*8, kernelSize = kernelSize, strd = strd, drop_me = True, drop_level = drpL)
	cn6 = tf.keras.layers.Concatenate()([deconvL6, convL4])
	#
	deconvL7 = getCLU(cn6,filterSize = filterSize*4, kernelSize = kernelSize, strd = strd, drop_me = True, drop_level = drpL)
	cn7 = tf.keras.layers.Concatenate()([deconvL7, convL3])
	#
	deconvL8 = getCLU(cn7,filterSize = filterSize*2, kernelSize = kernelSize, strd = strd, drop_me = False, drop_level = drpL)
	cn8 = tf.keras.layers.Concatenate()([deconvL8, convL2])
	#
	deconvL9 = getCLU(cn8,filterSize = filterSize, kernelSize = kernelSize, strd = strd, drop_me = False, drop_level = drpL)
	cn9 = tf.keras.layers.Concatenate()([deconvL9, convL1])
	#
	outLayer = tf.keras.layers.Conv2DTranspose(nChannels, kernelSize, strides = strd, padding=pdng, kernel_initializer="he_normal", activation=actFunc)(cn9) # use tanh, softsign, selu, elu
	return outLayer

### now the models
### Discriminator; a common discriminator for all generator models.
# TO DO: try to improve the discriminator models!

def Discriminator(input_shape = [128,128,1], target_shape = [128,128,5], nChannels = 5, kernel_size = 2, the_name = "mask2dapi_001"):
	inp = tf.keras.layers.Input(shape = input_shape, name = 'input_image')
	tar = tf.keras.layers.Input(shape = target_shape, name = 'target_image')
	#
	cn1 = tf.keras.layers.concatenate([inp, tar]) 
	cv1 = tf.keras.layers.Conv2D(64, kernel_size, strides = 2, padding='same', kernel_initializer="he_normal", use_bias=False)(cn1)
	ac1 = tf.keras.layers.LeakyReLU()(cv1)
	#
	cv2 = tf.keras.layers.Conv2D(128, kernel_size, strides = 2, padding='same', kernel_initializer="he_normal", use_bias=False)(ac1)
	bn2 = tf.keras.layers.BatchNormalization()(cv2)
	ac2 = tf.keras.layers.LeakyReLU()(bn2)
	#
	cv3 = tf.keras.layers.Conv2D(256, kernel_size, strides = 2, padding='same', kernel_initializer="he_normal", use_bias=False)(ac2)
	bn3 = tf.keras.layers.BatchNormalization()(cv3)
	ac3 = tf.keras.layers.LeakyReLU()(bn3)
	#
	zp4 = tf.keras.layers.ZeroPadding2D()(ac3)
	cv4 = tf.keras.layers.Conv2D(512, kernel_size, strides=1, kernel_initializer="he_normal", use_bias = False)(zp4)
	bn4 = tf.keras.layers.BatchNormalization()(cv4)
	ac4 = tf.keras.layers.LeakyReLU()(bn4)
	#
	zp5 = tf.keras.layers.ZeroPadding2D()(ac4)
	finalLayer = tf.keras.layers.Conv2D(nChannels, kernel_size, strides = 1, kernel_initializer = "he_normal")(zp5)
	model = tf.keras.Model(inputs = [inp, tar], outputs = finalLayer, name = the_name)
	#print(model.summary())
	return model

def discriminator_loss(loss_object,disc_real_output, disc_generated_output):
	real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
	generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
	total_disc_loss = real_loss + generated_loss
	return total_disc_loss

############### GENERATOR MODELS #######

# mask2dapi_001: 1x U-Net (filter_sizes: 16, 32, 64; kernel_sizes: 3,5,7; activation functions: tanh)
# mask2dapi_002: 2x U-Net (filter_sizes: 16; kernel_sizes: 3,5,7; activation functions: tanh, tanh or gelu, tanh)
# mask2dapi_003: 3x U_Net (filter_sizes: 16; kernel_sizes: 3,5,7; activation functions: tanh, tanh, tanh or tanh, gelu, tanh)

def mask2dapi_001(input_shape = (128,128,1), actFunc = "tanh", nChannels = 5, kernelSize = 3, filterSize = 16, strd = 2, drpL = 0.5, pdng = "same", useBs = False, unet = 6):
	input1 = tf.keras.layers.Input(input_shape)
	if unet == 6:
		outA1 = getUnet6(input1, nChannels = nChannels,actFunc = actFunc, filterSize = filterSize, kernelSize = kernelSize, strd = strd, pdng = pdng, useBs = useBs, drpL = drpL)
	else:
		outA1 = getUnet4(input1, nChannels = nChannels,actFunc = actFunc, filterSize = filterSize, kernelSize = kernelSize, strd = strd, pdng = pdng, useBs = useBs, drpL = drpL)
	model = tf.keras.Model(inputs = input1, outputs = outA1, name = inspect.stack()[0][3])
	return model

def mask2dapi_002a(input_shape = (128,128,1), actFunc = "tanh", nChannels = 5, kernelSize = 3, filterSize = 16, strd = 2, drpL = 0.5, pdng = "same", useBs = False, unet = 6):
	input1 = tf.keras.layers.Input(input_shape)
	#
	if unet == 6:
		outA1 = getUnet6(input1, nChannels = nChannels,actFunc = actFunc, filterSize = filterSize, kernelSize = kernelSize, strd = strd, pdng = pdng, useBs = useBs, drpL = drpL)
	else:
		outA1 = getUnet4(input1, nChannels = nChannels,actFunc = actFunc, filterSize = filterSize, kernelSize = kernelSize, strd = strd, pdng = pdng, useBs = useBs, drpL = drpL)
	inputA1 = tf.keras.layers.Concatenate()([input1, outA1])
	#
	if unet == 6:
		outA2 = getUnet6(inputA1, nChannels = nChannels, actFunc = actFunc, filterSize = filterSize, kernelSize = kernelSize, strd = strd, pdng = pdng, useBs = useBs, drpL = drpL)
	else:
		outA2 = getUnet4(inputA1, nChannels = nChannels, actFunc = actFunc, filterSize = filterSize, kernelSize = kernelSize, strd = strd, pdng = pdng, useBs = useBs, drpL = drpL)
	model = tf.keras.Model(inputs = input1, outputs = outA2, name = inspect.stack()[0][3])
	return model

def mask2dapi_002b(input_shape = (128,128,1), actFunc = "tanh", nChannels = 5, kernelSize = 3, filterSize = 16, strd = 2, drpL = 0.5, pdng = "same", useBs = False, unet = 6):
	input1 = tf.keras.layers.Input(input_shape)
	#
	if unet == 6:
		outA1 = getUnet6(input1, nChannels = nChannels,actFunc = "gelu", filterSize = filterSize, kernelSize = kernelSize, strd = strd, pdng = pdng, useBs = useBs, drpL = drpL)
	else:
		outA1 = getUnet4(input1, nChannels = nChannels,actFunc = "gelu", filterSize = filterSize, kernelSize = kernelSize, strd = strd, pdng = pdng, useBs = useBs, drpL = drpL)
	inputA1 = tf.keras.layers.Concatenate()([input1, outA1])
	#
	if unet == 6:
		outA2 = getUnet6(inputA1, nChannels = nChannels, actFunc = actFunc, filterSize = filterSize, kernelSize = kernelSize, strd = strd, pdng = pdng, useBs = useBs, drpL = drpL)
	else:
		outA2 = getUnet4(inputA1, nChannels = nChannels, actFunc = actFunc, filterSize = filterSize, kernelSize = kernelSize, strd = strd, pdng = pdng, useBs = useBs, drpL = drpL)
	model = tf.keras.Model(inputs = input1, outputs = outA2, name = inspect.stack()[0][3])
	return model

def mask2dapi_003a(input_shape = (128,128,1), actFunc = "tanh", nChannels = 5, kernelSize = 3, filterSize = 16, strd = 2, drpL = 0.5, pdng = "same", useBs = False, unet = 6):
	input1 = tf.keras.layers.Input(input_shape)
	#
	if unet == 6:
		outA1 = getUnet6(input1, nChannels = nChannels,actFunc = actFunc, filterSize = filterSize, kernelSize = kernelSize, strd = strd, pdng = pdng, useBs = useBs, drpL = drpL)
	else:
		outA1 = getUnet4(input1, nChannels = nChannels,actFunc = actFunc, filterSize = filterSize, kernelSize = kernelSize, strd = strd, pdng = pdng, useBs = useBs, drpL = drpL)
	inputA1 = tf.keras.layers.Concatenate()([input1, outA1])
	#
	if unet == 6:
		outA2 = getUnet6(inputA1, nChannels = nChannels, actFunc = actFunc, filterSize = filterSize, kernelSize = kernelSize, strd = strd, pdng = pdng, useBs = useBs, drpL = drpL)
	else:
		outA2 = getUnet4(inputA1, nChannels = nChannels, actFunc = actFunc, filterSize = filterSize, kernelSize = kernelSize, strd = strd, pdng = pdng, useBs = useBs, drpL = drpL)
	input2 = tf.keras.layers.Concatenate()([inputA1, outA2])
	#
	if unet == 6:
		outA3 = getUnet6(input2, nChannels = nChannels, actFunc = actFunc, filterSize = filterSize, kernelSize = kernelSize, strd = strd, pdng = pdng, useBs = useBs, drpL = drpL)
	else:
		outA3 = getUnet4(input2, nChannels = nChannels, actFunc = actFunc, filterSize = filterSize, kernelSize = kernelSize, strd = strd, pdng = pdng, useBs = useBs, drpL = drpL)
	model = tf.keras.Model(inputs = input1, outputs = outA3, name = inspect.stack()[0][3])
	return model

def mask2dapi_003b(input_shape = (128,128,1), actFunc = "tanh", nChannels = 5, kernelSize = 3, filterSize = 16, strd = 2, drpL = 0.5, pdng = "same", useBs = False, unet = 6):
	input1 = tf.keras.layers.Input(input_shape)
	#
	if unet == 6:
		outA1 = getUnet6(input1, nChannels = nChannels,actFunc = actFunc, filterSize = filterSize, kernelSize = kernelSize, strd = strd, pdng = pdng, useBs = useBs, drpL = drpL)
	else:
		outA1 = getUnet4(input1, nChannels = nChannels,actFunc = actFunc, filterSize = filterSize, kernelSize = kernelSize, strd = strd, pdng = pdng, useBs = useBs, drpL = drpL)
	inputA1 = tf.keras.layers.Concatenate()([input1, outA1])
	#
	if unet == 6:
		outA2 = getUnet6(inputA1, nChannels = nChannels, actFunc = "gelu", filterSize = filterSize, kernelSize = kernelSize, strd = strd, pdng = pdng, useBs = useBs, drpL = drpL)
	else:
		outA2 = getUnet4(inputA1, nChannels = nChannels, actFunc = "gelu", filterSize = filterSize, kernelSize = kernelSize, strd = strd, pdng = pdng, useBs = useBs, drpL = drpL)
	input2 = tf.keras.layers.Concatenate()([inputA1, outA2])
	#
	if unet == 6:
		outA3 = getUnet6(input2, nChannels = nChannels, actFunc = actFunc, filterSize = filterSize, kernelSize = kernelSize, strd = strd, pdng = pdng, useBs = useBs, drpL = drpL)
	else:
		outA3 = getUnet4(input2, nChannels = nChannels, actFunc = actFunc, filterSize = filterSize, kernelSize = kernelSize, strd = strd, pdng = pdng, useBs = useBs, drpL = drpL)
	model = tf.keras.Model(inputs = input1, outputs = outA3, name = inspect.stack()[0][3])
	return model

###
def generator_loss(loss_object, disc_generated_output, gen_output, target):
	gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
	l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
	total_gen_loss = gan_loss + (LAMBDA * l1_loss)
	return total_gen_loss, gan_loss, l1_loss

def getBatches(the_length, batch_size):
	dff = the_length % batch_size
	the_length -= dff
	idxs = []
	for idx in range(0, the_length, batch_size):
		idxs.append([idx,idx+batch_size])
	if dff != 0:
		idxs.append([the_length, the_length+dff])
	return idxs

def rotateImages(aa):
	print("Inverting ...")
	keep_aa = aa.copy()
	cc = keep_aa.copy()
	cc = cc.T
	cc = np.swapaxes(cc, 0,3)
	aa = np.concatenate((aa,cc), axis = 0)
	cc = keep_aa.copy()
	print("Rotating 90C - 1...")
	cc = np.rot90(cc,k = 1, axes = (1,2))
	aa = np.concatenate((aa,cc), axis = 0)
	cc = keep_aa.copy()
	print("Rotating 90C - 2 ...")
	cc = np.rot90(cc,k = 2, axes = (1,2))
	aa = np.concatenate((aa,cc), axis = 0)
	cc = keep_aa.copy()
	cc = np.rot90(cc,k = 3, axes = (1,2))
	aa = np.concatenate((aa,cc), axis = 0)
	return aa

def get_images(img, target_shape):
		#example_input = img[:,:,:,0:2]
		#example_target = img[:,:,:,2]
		example_input = img[:,:,:,1:]
		example_target = img[:,:,:,:1]
		#example_target = tf.reshape(example_target, (1,)+target_shape)
		return example_input, example_target

@tf.function
def train_step(the_generator, the_discriminator, loss_object, generator_optimizer, discriminator_optimizer, summary_writer, input_image, target, epoch):
	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
		gen_output = the_generator(input_image, training=True)
		disc_real_output = the_discriminator([input_image, target], training=True)
		disc_generated_output = the_discriminator([input_image, gen_output], training=True)
		gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(loss_object, disc_generated_output, gen_output, target)
		disc_loss = discriminator_loss(loss_object, disc_real_output, disc_generated_output)
	generator_gradients = gen_tape.gradient(gen_total_loss, the_generator.trainable_variables)
	discriminator_gradients = disc_tape.gradient(disc_loss, the_discriminator.trainable_variables)
	generator_optimizer.apply_gradients(zip(generator_gradients, the_generator.trainable_variables))
	discriminator_optimizer.apply_gradients(zip(discriminator_gradients, the_discriminator.trainable_variables))
	with summary_writer.as_default():
		tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
		tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
		tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
		tf.summary.scalar('disc_loss', disc_loss, step=epoch)

def fit_data(the_generator, the_discriminator, loss_object, generator_optimizer, discriminator_optimizer, summary_writer, checkpoint_prefix, train_ds, epochs, example_input, example_target, target_shape, checkpoint, batchSize = 8):
	idx = list(range(train_ds.shape[0]))
	random.shuffle(idx)
	for epoch in range(epochs):
		start = time.time()
		display.clear_output(wait=True)
		generate_images(the_generator, example_input, example_target, epoch)
		print("Epoch {} done on {}".format(epoch, time.strftime("%Y:%m:%d-%H:%M:%S")))
		# Train
		idxs = getBatches(train_ds.shape[0], batchSize)
		random.shuffle(idx)
		jj = 0
		for ii in idxs:
			if jj  % 5000 == 0:
				print("...")
			jj += 1
			input_image, target = get_images(train_ds[idx[ii[0]:ii[1]],...], target_shape)
			train_step(the_generator, the_discriminator, loss_object, generator_optimizer, discriminator_optimizer, summary_writer, input_image, target, epoch)
		if (epoch + 1) % 50 == 0:
			checkpoint.save(file_prefix = checkpoint_prefix)
			print ('{}; time taken for epoch {} is {} sec\n'.format(time.strftime("%Y:%m:%d-%H:%M:%S"), epoch + 1, time.time()-start))
		if (epoch + 1) % 25 == 0:
			the_generator.save("gnrt_{}_{}".format(the_generator.name, epoch+1))
			the_discriminator.save("dscr_{}_{}".format(the_discriminator.name, epoch+1))
	checkpoint.save(file_prefix = checkpoint_prefix)
	return the_generator, the_discriminator

##
def parse_masks2dapi_options():
	from optparse import OptionParser
	parser = OptionParser()
	parser.add_option("-R","--rangeValue", type = "int", default  = 1, dest = "rangeValue", help  = "range values, default: 1")
	parser.add_option("-a","--actFunc", type = "string", default  = "tanh", dest = "actFunc", help  = "activation function, default: tanh")
	parser.add_option("-b","--batchSize", type = "int", default  = 8, dest = "batchSize", help  = "batch size, inputSize: 8")
	parser.add_option("-i","--inputSize", type = "int", default  = 512, dest = "inputSize", help  = "the input pixel size, inputSize: 512")
	parser.add_option("-e","--epochs", type = "int", default  = 25, dest = "epochs", help  = "the number of epochs, default: 100")
	parser.add_option("-n","--nChannels", type = "int", default  = 5, dest = "nChannels", help  = "the number of channels, default: 5")
	parser.add_option("-k","--kernelSize", type = "int", default  = 3, dest = "kernelSize", help  = "kernel, default: 3")
	parser.add_option("-f","--filterSize", type = "int", default  = 16, dest = "filterSize", help  = "filter, default: 16")
	parser.add_option("-g","--theGPU2use", type = "int", default  = 0, dest = "theGPU2use", help  = "which GPU to use, default: 0")
	parser.add_option("-s","--strideSize", type = "int", default  = 2, dest = "strideSize", help  = "strideSize, default: 2")
	parser.add_option("-d","--dropLevel", type = "float", default  = 0.5, dest = "dropLevel", help  = "dropLevel, default: 0.5")
	parser.add_option("-u","--unet", type = "int", default  = 6, dest = "unet", help  = "unet 4 or 6 layers, default: 6")
	parser.add_option("-p","--padding", type = "string", default  = "same", dest = "padding", help  = "padding, default: same")
	parser.add_option("-B","--useBias", action =  "store_true", default  = False, dest = "useBias", help = "useBias, default: False ...")
	parser.add_option("-m","--modelName", type = "string", default  = "mask2dapi_001", dest = "modelName", help  = "modelName, default: mask2dapi_001")
	parser.add_option("-l","--mergeLayers", action =  "store_true", default  = False, dest = "mergeLayers", help = "merge layers, default: False ...")
	parser.add_option("-r","--rotate-images", action =  "store_true", default  = False, dest = "rotateImages", help = "rotate images, default: False ...")
	parser.add_option("-P","--train-data-path", type = "string", default  = None, dest = "trainPath", help = "the full path to the train data, default:  ...")
	return parser

def main(the_options):
	import sys
	parser = parse_masks2dapi_options()
	#the_options, args = parser.parse_args(["-lr","-P","/home/cosacakm/00python/Cosacaketal2022/createMasks2DapiTrainData","-i","128"])
	out = "m2d_unet"+str(the_options.unet)+"_".join([the_options.modelName, "shape",str(the_options.inputSize), "px_FS" , str(the_options.filterSize), "AF", the_options.actFunc, "KS", str(the_options.kernelSize),"ML", str(the_options.mergeLayers)])
	#
	if not os.path.exists(out):
		os.mkdir(out)
	#
	os.chdir(out)
	#
	setGPU(the_options.theGPU2use) # in case of many GPUs, use the first one.
	#
	div_norm = 127.5
	div_diff = 1.0
	#
	train_dataset = np.load(the_options.trainPath + "/mask2dapi_traindata_" + str(the_options.inputSize) + "px.npy")
	train_dataset = train_dataset[range(0, train_dataset.shape[0],the_options.rangeValue),40:-40,40:-40,:] # the training data has 40 black border. 
	if the_options.mergeLayers:
		train_dataset = train_dataset/1.0
		layer_merged = train_dataset[...,1:2]+train_dataset[...,2:3]+train_dataset[...,3:4]+train_dataset[...,4:5]
		layer_merged[layer_merged > 0] = 255
		train_dataset = np.concatenate((train_dataset[...,:1], layer_merged, train_dataset[...,1:]), axis = 3)
	#
	if the_options.rotateImages:
		train_dataset = rotateImages(train_dataset)
	#
	train_dataset = train_dataset/div_norm-div_diff
	#
	train_dataset = train_dataset.astype("float32")
	#idx = list(range(train_dataset.shape[0]))
	#random.shuffle(idx) # to shuffle train data, using TensorFlow shuffling may cause a memory error.
	#
	IMG_WIDTH = train_dataset.shape[1]
	IMG_HEIGHT = train_dataset.shape[2]
	#
	global LAMBDA
	LAMBDA = train_dataset.shape[0]
	#BUFFER_SIZE_TRAIN = LAMBDA*1
	#BATCH_SIZE_TRAIN = 1
	#
	test_dataset = np.load("../../testImages_512px.npy")
	if the_options.mergeLayers:
		test_dataset = test_dataset/1.0
		layer_test_merged = test_dataset[...,0:1]+test_dataset[...,1:2]+test_dataset[...,2:3]+test_dataset[...,3:4]
		layer_test_merged[layer_test_merged > 0] = 255
		test_dataset = np.concatenate((layer_test_merged, test_dataset), axis = 3)
	#
	if the_options.inputSize == 128:
		new_testdata = test_dataset[:,0:128,0:128,:]
		new_testdata = np.concatenate((new_testdata,test_dataset[:,128:256,128:256,:]), axis = 0)
		new_testdata = np.concatenate((new_testdata,test_dataset[:,128:256,128:256,:]), axis = 0)
		new_testdata = np.concatenate((new_testdata,test_dataset[:,256:384,256:384,:]), axis = 0)
		new_testdata = np.concatenate((new_testdata,test_dataset[:,384:512,384:512,:]), axis = 0)
		test_dataset = new_testdata.copy()
	elif the_options.inputSize == 256:
		new_testdata = test_dataset[:,0:256,0:256,:]
		new_testdata = np.concatenate((new_testdata,test_dataset[:,0:256,256:512,:]), axis = 0)
		new_testdata = np.concatenate((new_testdata,test_dataset[:,256:512,256:512,:]), axis = 0)
		new_testdata = np.concatenate((new_testdata,test_dataset[:,256:512,0:256,:]), axis = 0)
		test_dataset = new_testdata.copy()
	#
	#
	test_dataset = test_dataset/div_norm-div_diff
	tmp = np.zeros((test_dataset.shape[:3]+(1,)), dtype = test_dataset.dtype)
	test_dataset = np.concatenate((tmp, test_dataset), axis = 3)
	#
	BUFFER_SIZE_TEST = len(test_dataset)
	BATCH_SIZE_TEST = 1
	#
	test_target = test_dataset[:,:,:,0:1]
	test_dataset = test_dataset[:,:,:,1:]
	#
	OUTPUT_CHANNELS = 1
	input_shape = (IMG_HEIGHT, IMG_WIDTH, train_dataset.shape[3]-1)
	target_shape = (IMG_HEIGHT, IMG_WIDTH, 1)
	EPOCHS = the_options.epochs
	#
	loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
	if the_options.modelName == "mask2dapi_001":
		generator = mask2dapi_001(input_shape = input_shape, nChannels = OUTPUT_CHANNELS, kernelSize = the_options.kernelSize, filterSize = the_options.filterSize, unet = the_options.unet)
	elif the_options.modelName == "mask2dapi_002a":
		generator = mask2dapi_002a(input_shape = input_shape, nChannels = OUTPUT_CHANNELS, kernelSize = the_options.kernelSize, filterSize = the_options.filterSize, unet = the_options.unet)
	elif the_options.modelName == "mask2dapi_002b":
		generator = mask2dapi_002b(input_shape = input_shape, nChannels = OUTPUT_CHANNELS, kernelSize = the_options.kernelSize, filterSize = the_options.filterSize, unet = the_options.unet)
	elif the_options.modelName == "mask2dapi_003a":
		generator = mask2dapi_003a(input_shape = input_shape, nChannels = OUTPUT_CHANNELS, kernelSize = the_options.kernelSize, filterSize = the_options.filterSize, unet = the_options.unet)
	elif the_options.modelName == "mask2dapi_003b":
		generator = mask2dapi_003b(input_shape = input_shape, nChannels = OUTPUT_CHANNELS, kernelSize = the_options.kernelSize, filterSize = the_options.filterSize, unet = the_options.unet)
	else:
		raise ValueError("The Model Not Found, expected one of these  (mask2dapi_001, mask2dapi_002a, mask2dapi_002b, mask2dapi_003a, mask2dapi_003b), got {}".format(the_options.modelName))
	#
	discriminator = Discriminator(input_shape = input_shape, target_shape = target_shape, nChannels = OUTPUT_CHANNELS, the_name = the_options.modelName)
	#
	print(generator.summary())
	print(discriminator.summary())
	generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.9, beta_2 = 0.999)
	discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.9, beta_2 =0.999)
	checkpoint_dir = './training_checkpoints'
	checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
	checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, generator=generator, discriminator=discriminator)
	#
	if "training_checkpoints" in os.listdir("."):
		print("Loading the checkpoint ... : {}".format(time.strftime("%Y:%m:%d-%H:%M:%S")))
		checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
		print("Checkpoint loading done ... : {}".format(time.strftime("%Y:%m:%d-%H:%M:%S")))
		pass
	#
	#
	generate_images(generator, test_dataset, test_target, 0)
	#
	log_dir="logs/"
	#
	summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
	#
	q1,q2 = fit_data(generator, discriminator, loss_object,generator_optimizer, discriminator_optimizer, summary_writer, checkpoint_prefix, train_dataset, EPOCHS, test_dataset, test_target, target_shape, checkpoint, batchSize = the_options.batchSize)
	q1.save("gnrt_{}_{}".format(q1.name, EPOCHS))
	q2.save("dscr_{}_{}".format(q2.name, EPOCHS))

if __name__ == "__main__":
	import sys
	parser = parse_masks2dapi_options()
	the_options, args = parser.parse_args(sys.argv[1:])
	main(the_options)
