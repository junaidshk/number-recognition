import os
import sys
import shutil
import joblib
import numpy as np
from PIL import Image
from declarations import ROOT_DIR
from Model.model import predict_value,train_model
from Utility.process_image import extract_number
from flask import Flask, jsonify, request, render_template, make_response, Response

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(ROOT_DIR,"static")


# Using "POST" method to handle "POST" request sent
# by the client (our android application in this case)
@app.route('/model/predict', methods=['POST'])
def apicall():
	filename = ""

	# Check if the 'test_image' key is present in the files section of the request.
	# If it is that means that the android application (or whatever client sent this post request)
	# has successfully sent an image file (of the sketch of the digit)
	if 'test_image' in request.files:
		# Get that image file and store it in file variable

		file = request.files['test_image']

		# Also save the filename
		filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

		# Now save this file on EC2 instance storage
		file.save(filename)
	else:
		return (jsonify({"prediction":"No Prediction","filename":"No Filename"}))

	test_image_path = filename

	# Pass the path of this file which you just saved on EC2 Instance to your image processing function extract_number()
	# It will process it and return the a list of file paths (these filepaths are for images of digits extracted from
	# the original image).
	all_numbers = extract_number(test_image_path,os.path.join(app.config['UPLOAD_FOLDER'],filename.split('/')[-1]+"_"),"PROD")
	
	digits = all_numbers[0]
	X_coord = all_numbers[1]

	predictions = []

	for idx in np.argsort(X_coord):
		# Recognizing digits using the trained model on the local machine itself
		test_X = np.uint8(np.asarray(digits[idx]).reshape(1,-1))
		pred_Y = predict_value(test_X)[0]
		predictions.append(str(pred_Y))

	# Return result as response to the application
	return (jsonify({"prediction":''.join(predictions),"filename":filename}))

if __name__ == "__main__":
	print ("Add /model/predict at the end of the below given url to trigger the prediction module. Make sure you send a 'POST' request to it.")
	app.run('0.0.0.0',debug=True, port=5000)
