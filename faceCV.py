import cv2
import requests
import time
import json
import glob
import sys
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials

# Variables
Path = '/home/anu/PycharmProjects/ML/azure_face/anubhav'

KEY = '857253859e73497e90179a5bd8d95509'
ENDPOINT = "https://phase.cognitiveservices.azure.com/"

face_api_url = "https://centralasia.api.cognitive.microsoft.com/face/v1.0/detect"
headers = {'Content-Type': 'application/octet-stream', 'Ocp-Apim-Subscription-Key': KEY}
params = {'detectionModel': 'detection_01', 'returnFaceId': 'true', 'returnFaceRectangle': 'true', 'returnFaceAttributes': 'age, gender, emotion'}

GROUPS = []
PEOPLE = []
ID = []

face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

# Functions
def create_group(group):
    face_client.person_group.create(person_group_id=group, name=group)
    print('Created group {}'.format(group))

def create_person(person, group):
    globals()[person] = face_client.person_group_person.create(group, person)
    print('Person ID:', globals()[person].person_id)
    ID.append(globals()[person].person_id)

    photo_list = [file for file in glob.glob('*.jpg') if file.startswith(person)]
    time.sleep(1)
    for image in photo_list:
        face_client.person_group_person.add_face_from_stream(
            GROUPS[0], globals()[person].person_id, open(image, 'r+b'))
        print('Included photo {}'.format(image))
        time.sleep(1)

def train(group):
    print('Starting training for {}'.format(group))
    face_client.person_group.train(group)
    while True:
        training_status = face_client.person_group.get_training_status(group)
        print("Training status for {}: {}.".format(group, training_status.status))
        if training_status.status == 'succeeded':
            break
        elif training_status.status == 'failed':
            face_client.person_group.delete(person_group_id=group)
            sys.exit('Training the person group has failed.')
        time.sleep(5)

def setup():
    GROUPS.append(input('Define the group name -> ').lower())
    list(map(lambda x: create_group(x), GROUPS))

    people_list = []
    person_name = None
    while person_name != 'end':
        person_name = input("Enter the name of the person to associate with the group '{}' or enter 'end' to finish. -> ".format(GROUPS[0])).lower()
        if person_name != 'end':
            PEOPLE.append(person_name)
            people_list.append(person_name)

    if len(people_list) == 1:
        print('{} has been added to group {}'.format(PEOPLE[0], GROUPS[0]))
    else:
        last_name = people_list.pop()
        names = ', '.join(people_list)
        print('{} and {} have been added to group {}'.format(names, last_name, GROUPS[0]))

    list(map(lambda x: create_person(x, GROUPS[0]), PEOPLE))
    list(map(lambda x: train(x), GROUPS))

    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()
        image = cv2.imencode('.jpg', frame)[1].tobytes()

        response = requests.post(face_api_url, params=params, headers=headers, data=image)
        response.raise_for_status()
        faces = response.json()
        face_ids = [face['faceId'] for face in faces]

        global results
        for face in face_ids:
            results = face_client.face.identify(face_ids, GROUPS[0])

        # Getting landmarks
        for n, (face, person, id, name) in enumerate(zip(faces, results, ID, PEOPLE)):
            rect = face['faceRectangle']
            left, top = rect['left'], rect['top']
            right = int(rect['width'] + left)
            bottom = int(rect['height'] + top)

            draw_rect = cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
            att = face['faceAttributes']
            age = att['age']

            if len(person.candidates) > 0 and str(person.candidates[0].person_id) == str(id):
                print('Person for face ID {} is identified in {}.{}'.format(person.face_id, 'Frame', person.candidates[0].person_id))
                draw_text = cv2.putText(frame, 'Name: {}'.format(name), (left, bottom + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                faces[n]['name'] = str(name)
            else:
                draw_text = cv2.putText(frame, 'Name: Unknown', (left, bottom + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                faces[n]['name'] = 'Unknown'

        cv2.imshow('face_rect', draw_rect)

        k = cv2.waitKey(1) & 0xFF  # bitwise AND operation to get the last 8 bits
        if k == 27:
            print("Escape hit, closing...")
            break

def end(group_name):
    cv2.destroyAllWindows()
    face_client.person_group.delete(person_group_id=group_name)

# Start the code
setup()
# Stop and clean
end('group_name')