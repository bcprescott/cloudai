import cv2
import io
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials

def capture_frames():
     # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Return
    return img, gray, faces

def draw_local_rectangle(img, faces):
    for (x, y, w, h) in faces:
        try:
            rec = cv2.rectangle(img, (x, y), (x+w, y+h), (60, 169, 201), 2)
            cv2.putText(rec, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 3)
            cv2.putText(rec, str(confidence), (x+180, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 3)
            cv2.putText(rec, "Age"+str(age), (x+100, y+250), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 3)
        except NameError:
            rec = cv2.rectangle(img, (x, y), (x+w, y+h), (60, 169, 201), 2)
    return cv2.imshow('img', img)

def send_inference(gray, faces):
    global emotion
    global age
    global gender
    global confidence
    # Run inference
    stream = io.BytesIO(cv2.imencode('.jpg',gray)[1])
    output = face_client.face.detect_with_stream(stream,return_face_id=True,return_face_attributes=['age','gender','emotion'])
    emotion, confidence = get_emotion(output[0].face_attributes.emotion)
    age = output[0].face_attributes.age
    gender = output[0].face_attributes.gender
    
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
    emo_name = max(emoDict, key=emoDict.get)
    emo_level = emoDict[emo_name]
    return emo_name, emo_level

# Global Variables
KEY = "d8aa7548ef904e838eea77ec3adc20a2"
ENDPOINT = "https://c1-facerecog.cognitiveservices.azure.com/"
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))
cap = cv2.VideoCapture(1)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
frameNumber = 0

#Main Loop
while True:
    img, gray, faces = capture_frames()
    
    if frameNumber < 5:
        frameNumber = frameNumber + 1
    else:
        send_inference(gray,faces)
        frameNumber = 0

    draw_local_rectangle(img, faces)
    
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()