import base64

from flask import Flask, request, redirect, url_for, send_from_directory, send_file
from flask_env import MetaFlaskEnv
import json
import objects
import os
import requests
from google.cloud import vision

class Configuration(metaclass=MetaFlaskEnv):
    ENV_LOAD_ALL = False


FACE_FOLDER = 'faces/'

app = Flask(__name__)
app.config.from_object(Configuration)
client = vision.ImageAnnotatorClient()


subscription_key_face = "26e4d65134c648fe85bbb833d0fb5764"
assert subscription_key_face

face_api_url = 'https://westcentralus.api.cognitive.microsoft.com/face/v1.0/'
face_detect_url = face_api_url + "detect"
face_verify_url = face_api_url + "verify"

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="credentials/My-First-Project-c9d3bff47432.json"

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.after_request
def after_request(response):
    response.headers.add('Content-type', 'application/json')
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/api/analyse', methods=['POST'])
def analyze():
    post_body = request.get_json(force=True)
    localizer = objects.localize_objects(client, post_body['base64img'], post_body['height'], post_body['width'])
    return json.dumps(localizer)

@app.route('/api/calibrateCamera', methods=['POST'])
def calibrate():
    post_body = request.get_json(force=True)
    focalLength = objects.calibrate_camera(post_body['known_distance'], post_body['known_width'], post_body['base64img'])
    return json.dumps({'focal':focalLength})


@app.route('/api/face', methods=['POST'])
def face_recognition():
    post_body = request.get_json(force=True)
    if post_body['img_url']:
        imgdata = base64.b64decode(post_body['img_url'])
        filename = 'imagine.jpg'
        with open('faces/'+filename, 'wb') as f:
            f.write(imgdata)

        image_url = 'faces/' + filename
    else:
        return json.dumps({"Error": "Img_url is missing"})

    known_faces = [{'faceId': 'c45f5a20-6d25-43a2-a5b9-46c54a36d338',
                    'name': 'dinca'},
                   {'faceId': '21893291-fdc1-4b52-9351-9d989047cbd6',
                    'name': 'state'},
                   {'faceId': 'e156c0ff-80b2-451d-b943-7d7bc634780e',
                    'name': 'erik'},
                   ]
    image_data = open(image_url, "rb").read()

    headers = {'Ocp-Apim-Subscription-Key': subscription_key_face,
               'Content-Type': 'application/octet-stream'}

    params = {
        'returnFaceId': 'true',
        'returnFaceLandmarks': 'false',
        'returnFaceAttributes': '',
    }

    response = requests.post(face_detect_url, params=params, headers=headers, data=image_data)
    #return response.json()
    curr_face_id = response.json()[0]['faceId']
    #return json.dumps(response.json())
    headers = {'Ocp-Apim-Subscription-Key': subscription_key_face,
               'Content-Type': 'application/json'}

    index = 0
    identical = False
    while index < len(known_faces):
        params_verify = json.dumps({
            'faceId1': curr_face_id,
            'faceId2': known_faces[index]['faceId']
        }
        )
        response_verify = requests.post(face_verify_url, data=params_verify, headers=headers)
        resp_dict = response_verify.json()

        #return (json.dumps(resp_dict))
        if resp_dict['isIdentical'] == True:
            identical = True
            break
        index = index + 1

    if identical == True:
        person_name = known_faces[index]['name'].title()
    else:
        person_name = 'Stranger'

    return json.dumps({'Identical': identical, 'Name': person_name})

if __name__ == '__main__':
    app.run(host='0.0.0.0')
