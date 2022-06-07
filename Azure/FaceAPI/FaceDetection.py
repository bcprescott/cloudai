import cv2
import io
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials

def standard_stream():
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        try:
            rec = cv2.rectangle(img, (x, y), (x+w, y+h), (60, 169, 201), 1)
            cv2.putText(rec, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 3)
            cv2.putText(rec, str(confidence), (x+180, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 3)
            cv2.putText(rec, "Age:"+str(age), (x+100, y+250), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 3)
            cv2.imshow('img',img)
        except NameError:
            rec = cv2.rectangle(img, (x, y), (x+w, y+h), (60, 169, 201), 1)
    return

def send_inference():
    # Read the frame
    global emotion
    global age
    global gender
    global confidence
    global blur
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    try:
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # Run inference
        stream = io.BytesIO(cv2.imencode('.jpg',gray)[1])
        output = face_client.face.detect_with_stream(stream,return_face_id=True,return_face_attributes=['age','gender','emotion'])
        emotion, confidence = get_emotion(output[0].face_attributes.emotion)
        age = output[0].face_attributes.age
        gender = output[0].face_attributes.gender
        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            rec = cv2.rectangle(img, (x, y), (x+w, y+h), (60, 169, 201), 1)
            cv2.putText(rec, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 3)
            cv2.putText(rec, str(confidence), (x+180, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 3)
            cv2.putText(rec, "Age:"+str(age), (x+100, y+250), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 3)
            cv2.imshow('img',img)
    except IndexError:
        cv2.imshow('img', img)
    return



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

KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxx"
ENDPOINT = "https://{your face API service}.cognitiveservices.azure.com/"

face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FPS,60)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

frameNumber = 0

while True:
    if frameNumber < 5:
        standard_stream()
        frameNumber = frameNumber + 1
    else:
        send_inference()
        frameNumber = 0
    k = cv2.waitKey(10) & 0xff
    if k==27:
     break

cap.release()
cv2.destroyAllWindows()