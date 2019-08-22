import os
import cv2
import csv
import random
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt


def imageToPixel(imgPath):
	# imgPath = extract_number(imgPath)
	pixels = np.uint8(np.asarray(Image.open(imgPath)).reshape(1,-1))
	return pixels

# Can't use this on ec2. Only for Debugging Purposes
def show_image(img,msg,mode):
	if mode == "DEBUG":
		print(msg)
		cv2.imshow("dst",img)
		cv2.waitKey()

def show_image_of_region(img,region,msg,mode):
	if mode == "DEBUG":
		print(msg)
		x,y,w,h = region
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
		cv2.imshow("dst",img)
		cv2.waitKey()

def intersection(a,b):
	x = max(a[0], b[0])
	y = max(a[1], b[1])
	w = min(a[0]+a[2], b[0]+b[2]) - x
	h = min(a[1]+a[3], b[1]+b[3]) - y
	if w<0 or h<0: return (0,0,0,0)
	return (x, y, w, h)

def valid_region(rectangle_1,rectangle_2):

	brect_rectangle_1 = rectangle_1
	brect_rectangle_2 = rectangle_2

	if np.array_equal(brect_rectangle_1,brect_rectangle_2):
		return {"intersection":False,"current_region":False,"partial":False,"same":True,"rectangle_1":brect_rectangle_1,"rectangle_2":brect_rectangle_2}

	# cv2.rectangle(img_bw,(brect_rectangle_1[0],brect_rectangle_1[1]),(brect_rectangle_1[0]+brect_rectangle_1[2],brect_rectangle_1[1]+brect_rectangle_1[3]),(0,255,0),2)
	# show_image(img_bw,"rectangle_1_in_image","DEBUG")
	# cv2.rectangle(img_bw,(brect_rectangle_2[0],brect_rectangle_2[1]),(brect_rectangle_2[0]+brect_rectangle_2[2],brect_rectangle_2[1]+brect_rectangle_2[3]),(0,255,0),2)
	# show_image(img_bw,"rectangle_1_in_image","DEBUG")

	rectangle_3 = intersection(brect_rectangle_1,brect_rectangle_2)
	
	rectangle_1_area = brect_rectangle_1[2]*brect_rectangle_1[3]
	rectangle_2_area = brect_rectangle_2[2]*brect_rectangle_2[3]
	rectangle_3_area = rectangle_3[2]*rectangle_3[3]

	if rectangle_3_area != 0:
		if rectangle_1_area == rectangle_3_area:
			# print("Rectangle 1 in 2")
			return {"intersection":True,"current_region":True,"partial":False,"same":False,"rectangle_1":brect_rectangle_1,"rectangle_2":brect_rectangle_2}
		if rectangle_2_area == rectangle_3_area:
			# print("Rectangle 2 in 1")
			return {"intersection":True,"current_region":False,"partial":False,"same":False,"rectangle_1":brect_rectangle_1,"rectangle_2":brect_rectangle_2}
		
		if rectangle_1_area > rectangle_3_area and rectangle_2_area > rectangle_3_area:
			# print("Partial Overlap")
			if rectangle_1_area > rectangle_2_area:
				return {"intersection":True,"current_region":False,"partial":True,"same":False,"rectangle_1":brect_rectangle_1,"rectangle_2":brect_rectangle_2}
			else:
				return {"intersection":True,"current_region":True,"partial":True,"same":False,"rectangle_1":brect_rectangle_1,"rectangle_2":brect_rectangle_2}
	else:
		return {"intersection":False,"current_region":True,"partial":False,"same":False,"rectangle_1":brect_rectangle_1,"rectangle_2":brect_rectangle_2}

def remove_duplicates(_list):
	result = []
	duplicate_indexes = []
	for i in range(0,len(_list)-1):
		if i in duplicate_indexes:
			break
		for j in range(i+1,len(_list)):
			if _list[i] == _list[j]:
				duplicate_indexes.append(j)

	for i in range(0,len(_list)):
		if i not in duplicate_indexes:
			result.append(_list[i])

	return result

# MAIN FUNCTION 
def extract_number(imgPath, outputPath,mode="PROD"):
	digits = []

	#Reading the image file
	img_original = cv2.imread(imgPath)

	show_image(img_original,"Original Image",mode)

	# Change the Original Image to Black and white, if not already
	if img_original.shape[2] == 3:
		img_bw  = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
		img_bw = cv2.GaussianBlur(img_bw, (3, 3), 0)

	show_image(img_bw,"",mode)

	mser = cv2.MSER_create(_delta=2)
	mser.setMinArea(600)

	regions , coords = mser.detectRegions(img_bw)

	print("Processing {} Detected Regions".format(len(regions)))
	# valid_regions = regions
	valid_regions = [cv2.boundingRect(regions[0])]

	for current_region in regions[1:]:
		print("{} Valid Regions".format(len(valid_regions)))
		for region in valid_regions:
			vr = valid_region(region,cv2.boundingRect(current_region))
			print(vr)

			# When current region contains the previous valid region
			if (vr["intersection"] and vr["current_region"]):
				valid_regions.append(vr["rectangle_2"])
				valid_regions.remove(vr["rectangle_1"])
				valid_regions = remove_duplicates(valid_regions)

			# When current region is contained in the previous valid region
			if (vr["intersection"] and (not vr["current_region"])):
				if vr["rectangle_2"] in valid_regions:
					valid_regions.remove(vr["rectangle_2"])
				break

			# When the current region is not contained in any previous valid region
			if (not vr["intersection"]) and (not vr["same"]):
				valid_regions.append(vr["rectangle_2"])
				valid_regions = remove_duplicates(valid_regions)

	all_x = []

	for region in valid_regions:
		x,y,w,h = region

		all_x.append(x)
		
		cv2.rectangle(img_original,(x,y),(x+w,y+h),(0,255,0),2)
		show_image(img_original,"Original Image",mode)
		
		ret,thresh = cv2.threshold(img_bw,160,255,cv2.THRESH_BINARY)	
		cropped = cv2.bitwise_not(thresh[y:y+h,x:x+w])

		cv2.imwrite("x" + str(random.choice([1,2,5,4,6,9,89,3,67])) + ".png",cropped)
		
		show_image(cropped,"Cropped Digit from Black and White Image",mode)
		
		a = cropped.shape
		
		height = a[0]
		width = a[1]

		_x = height if height > width else width
		_y = height if height > width else width

		square= np.zeros((_x,_y), np.uint8)

		new_squared_image_height_component_1 = int((_y-height)/2)
		new_squared_image_height_component_2 = int(_y-(_y-height)/2)
		new_squared_image_width_component_1 = int((_x-width)/2)
		new_squared_image_width_component_2 = int(_x-(_x-width)/2)

		width_difference = (new_squared_image_width_component_2 - new_squared_image_width_component_1) - width
		new_squared_image_width_component_1 = new_squared_image_width_component_1 + width_difference

		height_difference = (new_squared_image_height_component_2 - new_squared_image_height_component_1) - height
		new_squared_image_height_component_1 = new_squared_image_height_component_1 + height_difference
		
		square[new_squared_image_height_component_1:new_squared_image_height_component_2, new_squared_image_width_component_1:new_squared_image_width_component_2] = cropped

		show_image(square,"Cropped Squared Image of the Digit",mode)

		square_length = square.shape[0]

		padding = int((4/20)*square_length)
		
		new_square = np.zeros((square_length + (2*padding),square_length + (2*padding)), np.uint8)
		new_square[padding:square_length+padding,padding:square_length+padding] = square

		show_image(new_square,"New Squared Image with padding",mode)

		kernel = np.ones((2,2), np.uint8)
		dilated_image = cv2.dilate(new_square,kernel)

		final_number = cv2.resize(dilated_image,(28,28),interpolation = cv2.INTER_AREA)

		kernel = np.ones((1,1), np.uint8)
		final_number = cv2.dilate(final_number,kernel)

		final_number [final_number > 0] = 255

		show_image(final_number,"Resized Squared Image with padding",mode)
		digits.append(final_number)

		

	return [digits,all_x]

def predict_value(test_X):
    basepath = ""

    # Load the already created model
    clf = joblib.load(basepath + 'model_rfc.pkl')

    # Predict on test data
    pred_Y = clf.predict(np.array(test_X))

    return pred_Y

if __name__ == '__main__':
	all_numbers = extract_number("testimage.jpg","./","DEBUG")
	
	digits = all_numbers[0]
	X_coord = all_numbers[1]

	predictions = []

	for idx in np.argsort(X_coord):
		# Recognizing digits using the trained model on the local machine itself
		test_X = np.uint8(np.asarray(digits[idx]).reshape(1,-1))
		pred_Y = predict_value(test_X)[0]
		predictions.append(str(pred_Y))

	print("The number is " + ''.join(predictions))