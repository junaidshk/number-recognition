# Number Recognition using ML & Image Processing.
Android App which uses an API to connect to a OpenCV &amp; ML model deployed on AWS, to detect the numbers captured in an image.
<br><br>
<b>Two Directories : </b><br>
<b>1. AWS_API</b><br>
<b>2. Android_App</b>
<br><br>
-> <b>AWS_API</b><br>
Consists of ML model, Flask API, Image processing code files that are to be deployed on AWS.<br>
<i><b>model.py</i></b> : ML Model building code file.<br>
<i><b>demo.py</i></b> : API file to process request and response from app to the image processing file.<br>
<i><b>process_image.py</i></b> : Image processing file to extract image and modify it as per the trained model.<br>
<i><b>model_rfc</i></b> : Saved trained model file, which is used by image processing file.<br>
<i><b>train</i></b> : Original data on which the ML model is trained.<br>
<br>
-><b>Android_App</b><br>
Consists of android application source code which is configured to interact with the flask api deployed on AWS.
<br><br><br>
<b><i>Note: If facing issue with configuring the android app, then we can check the working of ML model and image processing code directly using the process_image.py file. The file is setup for DEBUG purpose so it will show you all the stages of image processing.
I have provided test image, but you can use any image of your choice and name it as per the test image.
</i></b>
