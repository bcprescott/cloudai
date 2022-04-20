# Authors: Ben Prescott, Brandon Babcock
# Date: 4/20/2022
# 
# This is a simple Python script that uses OpenCV and the Azure FaceAPI service to 
# use your local webcam as a data source for detecting your sentiment (happy, sad, angry, confused, etc.). 
# 
# As you may expect, a streaming video is just a sequence of images (frames) that, when analyzed, can be 
# treated individually or as a sequence. The stream's frames per second (FPS) will deminish whenever an API
# call is made to the FaceAPI endpoint, as it will process and display the result back to the analyzed frame.
# 
# To help offset this, one simple method is to not send every frame for inference. Instead, we can only send
# every (n)th frame and persist the previous result until the next frame is sent to the API. This is the
# process we follow in this script. The current state will send every 5th frame to the FaceAPI service and
# will keep the last result displayed until a new result is received.
# 
# Things to changes/modify to your needs:
# 1. Azure FaceAPI auth variables (KEY, ENDPOINT)
# 2. Input device for cv2, set in 'cap = cv2.VideoCapture()'
# 3. 'frameOffset' variable, which tells it how many frames to process before sending for inference    

import cv2
import io
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials

# Primary function for capturing the frames
def capture_frames():
    # Read the individual frame from the source
    _, img = cap.read()
    # Converting to grayscale (may help with detection)
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Identify where faces are in the stream
    faces = face_cascade.detectMultiScale(grayimg, 1.1, 4)
    # Return some variables for use later
    return img, grayimg, faces

# Function for drawing the bounding box
def draw_local_rectangle(img, faces):
    for (x, y, w, h) in faces:
        try:
            # Draw a box and overlay the emotion, confidence score, and assumed age
            # Only runs if a frame had been sent for inference, as to persist the last result
            rec = cv2.rectangle(img, (x, y), (x+w, y+h), (60, 169, 201), 2)
            cv2.putText(rec, emotion, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (200, 200, 200), 2)
            cv2.putText(rec, str(confidence), (x+180, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (200, 200, 200), 2)
            cv2.putText(rec, "Age: "+str(age), (x+100, y+250), cv2.FONT_HERSHEY_COMPLEX, 0.9, (200, 200, 200), 2)
        except NameError:
            # Standard bounding box using only OpenCV
            rec = cv2.rectangle(img, (x, y), (x+w, y+h), (60, 169, 201), 2)
    return cv2.imshow('img', img)

# Function that sends a frame for inference to the FaceAPI service
def send_inference(grayimg, faces):
    global emotion
    global age
    global gender
    global confidence
    # Encode the image into bytes that can be read by the API
    try:
        stream = io.BytesIO(cv2.imencode('.jpg',grayimg)[1])
        # Identify the face and return attributes
        output = face_client.face.detect_with_stream(stream,return_face_id=True,return_face_attributes=['age','gender','emotion'])
        # Pull the emotion (happy/sad/etc) and the confidence score from the result
        emotion, confidence = get_emotion(output[0].face_attributes.emotion)
        # Get the expected age from the API
        age = output[0].face_attributes.age
        # Get the expected gender from the API
        gender = output[0].face_attributes.gender
    except IndexError:
        return grayimg

# Helper function that helps parse the FaceAPI results
def get_emotion(emoObject):
    emoDict = dict()
    emoDict['anger'] = emoObject.anger
    emoDict['contempt'] = emoObject.contempt
    emoDict['disgust'] = emoObject.disgust
    emoDict['fear'] = emoObject.fear
    emoDict['happy'] = emoObject.happiness
    emoDict['neutral'] = emoObject.neutral
    emoDict['sad'] = emoObject.sadness
    emoDict['surprised'] = emoObject.surprise
    # Get the emotion with the largest confidence score
    emo_name = max(emoDict, key=emoDict.get)
    # Return only that emotion 
    emo_level = emoDict[emo_name]
    return emo_name, emo_level


# Global Variables

# Define your Face API auth key/endpoint
KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxx"
ENDPOINT = "https://{FaceAPI-name}.cognitiveservices.azure.com/"

# Establish client for FaceAPI
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

# Set your input device number (0 for default webcam) for OpenCV 
cap = cv2.VideoCapture(0)

# Use built-in classifier to detect faces and bounding box
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Number of frames to process before sending to FaceAPI
frameOffset = 5

# Iterator variable for keeping tabs on the current frame count
# Do Not Change
currentFrame = 0


#Main Loop
while True:
    img, grayimg, faces = capture_frames()
    
    if currentFrame < frameOffset:
        currentFrame = currentFrame + 1
    else:
        send_inference(grayimg,faces)
        currentFrame = 0

    draw_local_rectangle(img, faces)
    
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()