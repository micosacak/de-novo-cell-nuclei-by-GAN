"""
Author: Mehmet Ilyas Cosacak
Year  : 2022.10.10
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import random
import scipy
import imageio

def readImages(prfx = None, borderSize = 40):
	if not prfx:
		raise ValueError(
	"""
	
	please give a prfx.
	
	this function assumes images in tif, png, or jpg format and is named
	with a prfx. The image's names must be ordered when sorted. The order
	must be dapi, mask layer 0, mask layer 1, mask layer 2, and mask layer 3
	edit the code below or write a new function to handle the images.
	
	"""
	)
	the_files = sorted([x for x in os.listdir() if (x.endswith(".tif") or x.endswith(".png") or x.endswith(".jpg")) and x.startswith(prfx)])
	the_image = [cv2.imread(x) for x in the_files]
	the_image = np.array(the_image)
	the_image = the_image/1.0 # convert to float64
	the_image[...,1:][the_image[...,1:] <= 150] = 0.0
	the_image[0,:,:,0] = the_image[0,:,:,0]
	the_image[1,:,:,0] = the_image[1,:,:,0]+the_image[1,:,:,1]+the_image[1,:,:,2]
	the_image[2,:,:,0] = the_image[2,:,:,0]+the_image[2,:,:,1]+the_image[2,:,:,2]
	the_image[3,:,:,0] = the_image[3,:,:,0]+the_image[3,:,:,1]+the_image[3,:,:,2]
	the_image[4,:,:,0] = the_image[4,:,:,0]+the_image[4,:,:,1]+the_image[4,:,:,2]
	the_image = np.swapaxes(the_image,0,3)
	the_image = the_image[0,...]
	the_image[...,1:][the_image[...,1:] > 0] = 255
	# add zero paddings with 40 pixel to images
	# this will prevent mis location of masks add the border.
	theBorder = np.zeros((borderSize,the_image.shape[1], the_image.shape[2]))
	the_image = np.concatenate((theBorder, the_image,theBorder), axis = 0)
	theBorder = np.zeros((the_image.shape[0],borderSize, the_image.shape[2]))
	the_image = np.concatenate((theBorder, the_image,theBorder), axis = 1)
	return the_image.copy()

# if the masks to be shown have 4 layers
# better to convert the 4 th layer masks to a gray color
def To3Chanels(x):
	x[...,0][x[...,3] > 0] = 220
	x[...,1][x[...,3] > 0] = 220
	x[...,2][x[...,3] > 0] = 220
	return x[...,:3].copy()

# a simple function to show images for visual inspection
def pp(x):
	plt.imshow(x)
	plt.show()

# check the masks
# pp(To3Chanels(the_images[0][...,1:].copy()))

# use cv2.findContours to get individual masks
# sometimes, unseen masking may prevent proper identification of each mask.
# in that case, the unseen masks must be erased.

# use the function below to get the contours from layers.
# use CHAIN_APPROX_NONE instead of CHAIN_APPROX_SIMPLE
def getContours(orgImage, theMin = 127):
	if orgImage.max() <= 1.0:
		orgImage = orgImage*255.0
	if orgImage.dtype != "uint8":
		orgImage = orgImage.astype("uint8")
	ret, thresh = cv2.threshold(orgImage, theMin, 255, 0)
	theConts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	return theConts

#

# it might be better to have layers in lists
# thus if one is not sure to put a mask on one of the 4 layers,
# the mask can be put on the 5th or forth layers.
# then a custom function (getLayers, see below) can be used to arrange masks on the layers
# the_contours = []
# for layer in layers:
# 	theConts = getContours(layer, theMin = 127.5)
# 	the_contours.extend(theConts)

#
def getImg(tmp, the_dif = 20, the_stp = 3, dimSizeX = 512, dimSizeY = 512):
	i_dif = int((tmp[...,1].max()-tmp[...,1].min())/2)+tmp[...,1].min()
	j_dif = int((tmp[...,0].max()-tmp[...,0].min())/2)+tmp[...,0].min()
	i0 = i_dif - the_dif
	i1 = i_dif + the_dif
	j0 = j_dif - the_dif
	j1 = j_dif + the_dif
	if i0 < 0:
		i0 = 0
	#
	if j0 < 0:
		j0 = 0
	#
	if i1 > dimSizeX:
		i1 = dimSizeX
	#
	if j1 > dimSizeY:
		j1 = dimSizeY
	aa = np.zeros((i1-i0,j1-j0))
	bb = tmp.copy()
	bb[...,0] = bb[...,0]-j_dif+the_dif
	bb[...,1] = bb[...,1]-i_dif+the_dif
	the_idxs = extendTheContours(bb, the_stp = the_stp)
	cc1 = cv2.fillPoly(aa.copy(),[bb],(255,255,255))
	cc2 = cv2.fillPoly(aa.copy(),[the_idxs],(255,255,255))
	cc1 = cc1/255
	cc2= cc2/255
	return i0,i1,j0,j1,cc1.copy(),cc2.copy()

def extendTheContours(the_contour, the_stp = 3):
	the_idxs = []
	for idx in the_contour:
		#print(idx)
		idx0 = idx[0]
		idx1 = idx[1]
		for ii in range(-the_stp,the_stp):
			the_idxs.append([idx0+ii,idx1])
			the_idxs.append([idx0+ii,idx1+1])
			the_idxs.append([idx0+ii,idx1-1])
			the_idxs.append([idx0+ii,idx1+2])
			the_idxs.append([idx0+ii,idx1-2])
			#
			the_idxs.append([idx0,idx1+ii])
			the_idxs.append([idx0+1,idx1+ii])
			the_idxs.append([idx0-1,idx1+ii])
			the_idxs.append([idx0+2,idx1+ii])
			the_idxs.append([idx0-2,idx1+ii])
	the_idxs = np.array(the_idxs)
	the_idxs[the_idxs < 0] = 0
	return  the_idxs

# TO DO: this function may generate same masks, write a function to remove redundant images to decrease running time
def getLayers(theImage, shuffleMe = False, the_contours = None, the_dif = 30, theMin = 50, the_stp = 4, maxCell0 = 800, maxCell1 = 400, maxCell2 = 200, maxCell3 = 100, theMinStp = None, tryNtimes = 10000):
	the_img = theImage.copy()
	dimSizeX = the_img.shape[0]
	dimSizeY = the_img.shape[1]
	tmp_image = np.zeros(((dimSizeX, dimSizeY,the_img.shape[2])))
	#
	if not theMinStp:
		theMinStp = the_stp-1
	tmp_contours = []
	for i in range(0,the_img.shape[2]):
		tmp_contours.extend(getContours(the_img[...,i].copy(), theMin = theMin))
	#
	if shuffleMe:
		random.shuffle(tmp_contours)
	#
	#print(len(tmp_contours))
	#
	reRun = True
	hh = 0
	getIt = 1 # return 1 if the masks could be arranged, else 0
	while reRun:
		hh += 1
		if hh >= tryNtimes:
			hh = 0
			the_stp -= 1
		if the_stp <= theMinStp:
			getIt = 0
			break
		new_image = tmp_image.copy()
		layer_0_counts = 0
		layer_1_counts = 0
		layer_2_counts = 0
		layer_3_counts = 0
		for k in range(len(tmp_contours)):
			i0,i1,j0,j1,contour1,contour2 = getImg(tmp_contours[k][:,0,:].copy(), the_dif = the_dif, the_stp = the_stp, dimSizeX = dimSizeX, dimSizeY = dimSizeY)
			layer_0 = new_image[i0:i1,j0:j1,0]
			layer_1 = new_image[i0:i1,j0:j1,1]
			layer_2 = new_image[i0:i1,j0:j1,2]
			layer_3 = new_image[i0:i1,j0:j1,3]
			if not np.any(layer_0+contour2 > 1) and layer_0_counts <= maxCell0:
				new_image[i0:i1,j0:j1,0] = new_image[i0:i1,j0:j1,0]+contour1.copy()
				layer_0_counts += 1
				reRun = False
			else:
				if not np.any(layer_1+contour2 > 1) and layer_1_counts <= maxCell1:
					new_image[i0:i1,j0:j1,1] = new_image[i0:i1,j0:j1,1]+contour1.copy()
					layer_1_counts += 1
					reRun = False
				else:
					if not np.any(layer_2+contour2 > 1) and layer_2_counts <= maxCell2:
						new_image[i0:i1,j0:j1,2] = new_image[i0:i1,j0:j1,2]+contour1.copy()
						layer_2_counts += 1
						reRun = False
					else:
						if not np.any(layer_3+contour2 > 1) and layer_3_counts <= maxCell3:
							new_image[i0:i1,j0:j1,3] = new_image[i0:i1,j0:j1,3]+contour1.copy()
							layer_3_counts += 1
							reRun = False
						else:
							print("Could not put the mask at index {}".format(k))
							reRun = False
							#break #TO DO: better to break here! 
		print("*"*20)
		random.shuffle(tmp_contours)
	return new_image, getIt

### now lets split images
def getMaxDiv(x_org,targetSize  = 512):
	if x_org % targetSize  == 0:
		x_new = copy.copy(x_org)
	else:
		x_new = (int(x_org/targetSize )+1)*targetSize
	return x_new

def getIndexes(imgHeight, imgWidth, targetSize = 512, overlap = 20):
	idx = []
	for i in range(imgHeight):
		for j in range(imgWidth):
			if i < imgHeight -1:
				h0 = i*targetSize
				h1 = (i+1)*targetSize+overlap
				if j < imgWidth-1:
					w0 = j*targetSize
					w1 = (j+1)*targetSize+overlap
				else:
					w0 = j*targetSize-overlap
					w1 = (j+1)*targetSize
			else:
				h0 = i*targetSize-overlap
				h1 = (i+1)*targetSize
				if j < imgWidth-1:
					w0 = j*targetSize
					w1 = (j+1)*targetSize+overlap
				else:
					w0 = j*targetSize-overlap
					w1 = (j+1)*targetSize
			idx.append((h0,h1,w0,w1))
	return idx

def splitImage(img, targetSize = 256, overlap = 256):
	imgHeight = int(getMaxDiv(img.shape[0], targetSize = targetSize)/targetSize)
	imgWidth  =  int(getMaxDiv(img.shape[1], targetSize = targetSize)/targetSize)
	idxs = getIndexes(imgHeight, imgWidth, targetSize = targetSize, overlap = overlap)
	tmp_image = np.zeros((imgHeight*targetSize, imgWidth*targetSize,img.shape[2])).astype("uint8")
	idx0 = int((tmp_image.shape[0]-img.shape[0])/2)
	idx1 = int((tmp_image.shape[1]-img.shape[1])/2)
	tmp_image[idx0:img.shape[0]+idx0, idx1:img.shape[1]+idx1,:] = img.copy()
	batchImages = np.zeros((len(idxs), targetSize+overlap, targetSize+overlap, img.shape[2])).astype("uint8")
	for ii in range(len(idxs)):
		#print(ii, idxs[ii])
		batchImages[ii,:,:,:] = tmp_image[idxs[ii][0]:idxs[ii][1],idxs[ii][2]:idxs[ii][3],:].copy()
	return batchImages

def parse_masks2dapi_options():
	from optparse import OptionParser
	parser = OptionParser()
	parser.add_option("-p","--imagesPrefixes", type = "string", default  = ["img_00","img_11"], dest = "imagesPrefixes", help  = "the prefixes of the images by comma separated, default: 'img_00,mg_11'")
	parser.add_option("-d","--dimSizeTarget", type = "int", default  = 512, dest = "dimSizeTarget", help  = "the target size of the images, default: 512")
	parser.add_option("-b","--borderSize", type = "int", default  = 40, dest = "borderSize", help  = "the border size to zero padding to images, default: 40")
	parser.add_option("-n","--nImages", type = "int", default  = 10, dest = "nImages", help  = "the number of randomly distributed masks to be generated per image, default: 10")
	parser.add_option("-N","--tryNtimes", type = "int", default  = 10000, dest = "tryNtimes", help  = "how many times to re-try to put all masks for images, default: 10000")
	parser.add_option("-s","--minSpace", type = "int", default  = 2, dest = "minSpace", help  = "the minimum space/distance between the masks to be on the same layer, default: 2 pixel")
	return parser

# TO DO: add these options to parse_masks2dapi_options, mainly used in getLayers function.
# the_dif = 30
# theMin = 50
# the_stp = 4
# maxCell0 = 800
# maxCell1 = 400
# maxCell2 = 200
# maxCell3 = 100
# theMinStp = None

def tm():
	return time.strftime("%Y:%m:%d - %H:%M:%S")


### 
runMe = True
if runMe:
	the_options = {
	"nImages":10, "borderSize":40, "dimSizeTarget":512,"imagesPrefixes":"img_00,img_11","tryNtimes":20,"minSpace":2
	}
	print(the_options)
	nn = the_options["nImages"]
	borderSize = the_options["borderSize"]
	dimSizeTarget = the_options["dimSizeTarget"]
	the_prefixes = the_options["imagesPrefixes"].split(",")
	tryNtimes = the_options["tryNtimes"]
	minSpace = the_options["minSpace"]
	the_images = []
	for prfx in the_prefixes: 
		the_images.append(readImages(prfx = prfx, borderSize = borderSize))
	#
	new_images = []
	for the_image in the_images:
		img_copy = the_image.copy()
		the_new_layers, TrueFalse = getLayers(the_image[...,1:].copy(), tryNtimes = tryNtimes, the_stp = minSpace)
		# pp(To3Chanels(the_new_layers.copy()))xs
		if TrueFalse != 1:
			raise ValuError("the masks could not be generated!")
		the_new_layers = the_new_layers * 255
		the_image = np.concatenate((the_image[...,:1], the_new_layers), axis = 2)
		new_images.append(the_image.copy())
		for i in range(10):
			the_image = img_copy.copy()
			the_new_layers, TrueFalse = getLayers(the_image[...,1:].copy(), shuffleMe = True, tryNtimes = tryNtimes, the_stp = minSpace)
			# pp(To3Chanels(the_new_layers.copy()))xs
			if TrueFalse != 1:
				raise ValuError("the masks could not be generated!")
			the_new_layers = the_new_layers * 255
			the_image = np.concatenate((the_image[...,:1], the_new_layers), axis = 2)
			new_images.append(the_image.copy())
			
if not os.path.exists("figure_1"):
	os.mkdir("figure_1")


def changeColors(tmp):
	tt = tmp.copy()
	tt = tt[...,1:]
	cc = tt.copy()
	tt[...,0][cc[...,0] > 0] = 100
	tt[...,1][cc[...,0] > 0] = 125
	tt[...,2][cc[...,0] > 0] = 180
	tt[...,0][cc[...,1] > 0] = 125
	tt[...,1][cc[...,1] > 0] = 60
	tt[...,2][cc[...,1] > 0] = 125
	tt[...,0][cc[...,2] > 0] = 125
	tt[...,1][cc[...,2] > 0] = 125
	tt[...,2][cc[...,2] > 0] = 125
	tt[...,0][cc[...,3] > 0] = 220
	tt[...,1][cc[...,3] > 1] = 220
	tt[...,2][cc[...,3] > 2] = 220
	return tt[...,:3].astype("uint8")

def To3Chanels(x):
	x[...,0][x[...,3] > 0] = 220
	x[...,1][x[...,3] > 0] = 220
	x[...,2][x[...,3] > 0] = 220
	return x[...,:3].copy()


for i in range(len(new_images)):
	tmp = new_images[i]
	if i == 0:
		aa  = np.concatenate((tmp[...,0:1],tmp[...,0:1],tmp[...,0:1]), axis = 2)
		aa = aa.copy()
		aa[...,1:3] = 0
		cv2.imwrite("figure_1/img_00_the_dapi.png", aa)
	if i == 11:
		aa  = np.concatenate((tmp[...,0:1],tmp[...,0:1],tmp[...,0:1]), axis = 2)
		aa = aa.copy()
		aa[...,1:3] = 0
		cv2.imwrite("figure_1/img_11_the_dapi.png", aa)
	if i < 11:
		cv2.imwrite("figure_1/img_00_{:04d}.png".format(i), To3Chanels(tmp[...,1:])) #changeColors(tmp))
	if i > 11:
		cv2.imwrite("figure_1/img_11_{:04d}.png".format(i), To3Chanels(tmp[...,1:])) #changeColors(tmp))
	

def main(the_options):
	print(the_options)
	nn = the_options.nImages
	borderSize = the_options.borderSize
	dimSizeTarget = the_options.dimSizeTarget
	the_prefixes = the_options.imagesPrefixes.split(",")
	tryNtimes = the_options.tryNtimes
	minSpace = the_options.minSpace
	the_images = []
	for prfx in the_prefixes: #["img_00","img_11"]:
		the_images.append(readImages(prfx = prfx, borderSize = borderSize))
	#
	new_images = []
	for the_image in the_images:
		the_new_layers, TrueFalse = getLayers(the_image[...,1:].copy(), tryNtimes = tryNtimes, the_stp = minSpace)
		# pp(To3Chanels(the_new_layers.copy()))xs
		if TrueFalse != 1:
			raise ValuError("the masks could not be generated!")
		the_new_layers = the_new_layers * 255
		the_image = np.concatenate((the_image[...,:1], the_new_layers), axis = 2)
		new_images.append(the_image.copy())
	#
	#np.save("original_images.npy",new_images)
	#
	splitedImages = np.zeros((1,dimSizeTarget,dimSizeTarget,5)) # an empty image
	for i in range(len(new_images)):
		the_image = new_images[1]
		splts = splitImage(the_image, targetSize = int(dimSizeTarget/2), overlap = int(dimSizeTarget/2))
		for ii in range(0,int(dimSizeTarget/2),4):
			splt0 = splitImage(the_image, targetSize = dimSizeTarget-ii, overlap = ii)
			splts = np.concatenate((splts, splt0), axis = 0)
		splitedImages = np.concatenate((splitedImages, splts.copy()), axis = 0)	
	#
	theBorder = np.zeros((splitedImages.shape[0],borderSize,splitedImages.shape[2], splitedImages.shape[3]))
	splitedImages = np.concatenate((theBorder, splitedImages,theBorder), axis = 1)
	theBorder = np.zeros((splitedImages.shape[0],splitedImages.shape[1],borderSize, splitedImages.shape[3]))
	splitedImages = np.concatenate((theBorder, splitedImages,theBorder), axis = 2)
	splitedImages = splitedImages[1:,...] # be careful to remove images with no masks
	#
	newImggs = np.zeros((splitedImages.shape[0]*nn, splitedImages.shape[1], splitedImages.shape[2], splitedImages.shape[3]), dtype = "uint8")
	#
	print("Now arranging the masks ...")
	newFlls = []
	for ii in range(splitedImages.shape[0]):
		print(ii)
		tmp_image = splitedImages[ii,...].copy()
		yzs = [getLayers(tmp_image[...,1:].copy(), shuffleMe = True, the_stp = minSpace, tryNtimes = tryNtimes) for i in range(nn)]
		yy = [np.concatenate((tmp_image[...,:1], xx[0]*255), axis = 2) for xx in yzs]
		yy = np.array(yy)
		yy = yy.astype("uint8")
		newImggs[(ii*nn):(ii+1)*nn,...] = yy.copy()
		zz = [xx[1] for xx in yzs]
		newFlls.extend(zz)
	#
	newFlls = np.array(newFlls).astype("uint8")
	np.save("mask2dapi_traindata_"+str(dimSizeTarget) + "px.npy", newImggs)
	np.save(prfx + "_" + str(dimSizeTarget) + "px_newFlls.npy", newFlls)
	#
	print("... done ...")
	xx = [the_images[0][...,1:].copy()]
	yy = [getLayers(the_images[0][...,1:].copy(), shuffleMe = True, the_stp = minSpace, tryNtimes = tryNtimes)[0]*255 for i in range(20)]
	xx.extend(yy)
	xx = [To3Chanels(x.astype("uint8")) for x in xx]
	imageio.mimsave("test.gif",xx, fps = 1)

if __name__ == "__main__":
	import sys
	parser = parse_masks2dapi_options()
	options, args = parser.parse_args(sys.argv[1:])
	main(options)


