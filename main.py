# import the necessary packages
from imutils.video import VideoStream
from imutils import paths
import itertools
import argparse
import imutils
import time
import cv2
import pyautogui

#backframe=cv2.imread("")                                                    ###path of the frame
#backframe=imutils.resize(backframe,width=600,height=600)
noi=0                                                                       ### no. of images

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--models", required=True,
	help="path to directory containing neural style transfer models")
args = vars(ap.parse_args())


modelPaths = paths.list_files(args["models"], validExts=(".t7",))
modelPaths = sorted(list(modelPaths))

# generate unique IDs for each of the model paths, then combine the
# two lists together
models = list(zip(range(0, len(modelPaths)), (modelPaths)))


modelIter = itertools.cycle(models)
(modelID, modelPath) = next(modelIter)


net = cv2.dnn.readNetFromTorch(modelPath)


cam=cv2.VideoCapture(0)

path="/home/the_confused_1/roboism/ProjectPics/image"                  #you can define the path to save pics according to your machine


# loop over frames from the video file stream
while cam.isOpened():
	# grab the frame from the threaded video stream
	ret,frame = cam.read()

	# resize the frame to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
	frame = imutils.resize(frame, width=600,height=600)
	orig = frame.copy()
	(h, w) = frame.shape[:2]

	# construct a blob from the frame, set the input, and then perform a
	# forward pass of the network
	blob = cv2.dnn.blobFromImage(frame, 1.0, (w, h),
		(103.939, 116.779, 123.680), swapRB=False, crop=False)
	net.setInput(blob)
	output = net.forward()

	# reshape the output tensor, add back in the mean subtraction, and
	# then swap the channel ordering
	output = output.reshape((3, output.shape[2], output.shape[3]))
	output[0] += 103.939
	output[1] += 116.779
	output[2] += 123.680
	output /= 255.0
	output = output.transpose(1, 2, 0)

	# show the original frame along with the output neural style
	# transfer
	cv2.imshow("Input", frame)
	cv2.imshow("Output", output)
	key = cv2.waitKey(1) & 0xFF

	# if the `n` key is pressed (for "next"), load the next neural
	# style transfer model
	if key == ord("n"):
		# grab the next nueral style transfer model model and load it
		(modelID, modelPath) = next(modelIter)
		print("[INFO] {}. {}".format(modelID + 1, modelPath))
		net = cv2.dnn.readNetFromTorch(modelPath)

	# otheriwse, if the `q` key was pressed, break from the loop
	elif key == ord("q"):
		break

	elif key==ord("c"):
		pic=pyautogui.screenshot()
		print(type(pic))
		noi=noi+1
		pic.save(path+str(noi)+".jpg")
		res=cv2.imread(path+str(noi)+".jpg")
		res=res[50:550,50:650]
		print("saving........")
		#res=cv2.addWeighted()
		cv2.imwrite(path+str(noi)+".jpg",res)

# do a bit of cleanup
cv2.destroyAllWindows()
cam.release()
