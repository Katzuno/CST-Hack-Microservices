import imutils as imutils
from google.cloud import vision
import base64
import cv2
import numpy as np

FOCAL_LENGTH = 55

OBJECTS_HEIGHT = {
	'laptop' : 30,
    'computer': 30,
	'person' : 50,
	'bottle' : 20,
	'table' : 90,
	'chair' : 110,
	'phone' : 15,
	'paper' : 11,
	'mouse' : 5,
	'glasses' : 5,
	'sunglasses' : 6,
	'jeans' : 40,
	'man' : 50,
	'woman' : 160,
	'desk' : 110
}

class Object:
    def __init__(self, vertex, name, confidence, img_height = 400, img_width = 400):
        self.vertex = vertex
        self.direction = None
        self.confidence = confidence
        self.name = name.lower()
        self.area = None
        self.real_height = None
        self.img_height = img_height
        self.distance = None
        self.obj_height = (self.vertex[2].y - self.vertex[1].y) * img_height
        self.obj_width = (self.vertex[1].x - self.vertex[0].x) * img_width
        self.compute_objects_height()
        self.calculate_distance()

    def compute_objects_height(self):
        #print(self.name)
        if self.name in OBJECTS_HEIGHT.keys():
            self.real_height = OBJECTS_HEIGHT[self.name]
        else:
            self.real_height = 15

    def calculate_distance(self):
        #print(self.img_height)
        self.distance = (FOCAL_LENGTH * self.real_height) / (self.obj_width)

    def get_directions(self):
        center_w = 0.5
        error = 0.2
        x_center = (self.vertex[3].x + self.vertex[2].x) / 2

        if x_center <= center_w - error:
            self.direction = 'left'
        elif x_center > center_w + error:
            self.direction = 'right'
        else:
            self.direction = 'front'

    def get_area(self):
        length = self.vertex[1].x - self.vertex[0].x
        width = self.vertex[3].y - self.vertex[0].y
        self.area = length * width

    def to_json(self):
        json_resp = {
            'name' : self.name,
            'direction' : self.direction,
            'area' : self.area,
            'confidence' : self.confidence,
            'distance' : self.distance,
            'real_height' : self.real_height,
            'vertex' : {
                'x1' : self.vertex[0].x, 'y1' : self.vertex[0].y,
                'x2' : self.vertex[1].x, 'y2' : self.vertex[1].y,
                'x3' : self.vertex[2].x, 'y3' : self.vertex[2].y,
                'x4' : self.vertex[3].x, 'y4' : self.vertex[3].y
            }
        }
        return json_resp

    def print_obj(self):
        print("Name: ", self.name, " || ", "Direction: ", self.direction)


def localize_objects(client, base64string, image_height = 400, image_width = 400):
    """Localize objects in the local image.

    Args:
    path: The path to the local file.
    """

    content = base64.b64decode(base64string)
    #print(content)
    # with open('images/test-photo.jpeg', 'rb') as image_file:
    #    content = image_file.read()
    image = vision.types.Image(content=content)

    objects = client.object_localization(image=image).localized_object_annotations

    #print('Number of objects found: {}'.format(len(objects)))

    response_objects = []
    max1 = 0
    index = 0
    for obj in objects:
        detected_object_formatted = Object(obj.bounding_poly.normalized_vertices, obj.name, obj.score, image_height, image_width)
        detected_object_formatted.get_directions()
        detected_object_formatted.get_area()


        print('=====', obj.name, ' ==========', calibrate_camera(detected_object_formatted.obj_width, 1000, 1750))
        # if max1 < detected_object_formatted.area:
        #    max1 = detected_object_formatted


        if detected_object_formatted.area:
            response_objects.append(detected_object_formatted.to_json())

    return response_objects


def readb64(encoded_data):
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img


def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth

def find_marker(image):
    # convert the image to grayscale, blur it, and detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 35, 125)

    # find the contours in the edged image and keep the largest one;
    # we'll assume that this is our piece of paper in the image
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    print(cv2.minAreaRect(c))
    # compute the bounding box of the of the paper region and return it
    return cv2.minAreaRect(c)

def calibrate_camera(object_width, known_distance, known_width):
    focalLength = (object_width * known_distance) / known_width

    return focalLength
