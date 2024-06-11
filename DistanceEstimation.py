import cv2 as cv 
import numpy as np

# Distance constants 
KNOWN_DISTANCE = 25 #INCHES
PERSON_WIDTH = 16 #INCHES
MOBILE_WIDTH = 3.0 #INCHES
MONITOR_WIDTH = 16
BOTTLE_WIDTH = 2.8
CHAIR_WIDTH = 13
LAPTOP_WIDTH = 10
KEYBOARD_WIDTH = 9
MOUSE_WIDTH = 3    
BAG_WIDTH = 12
PEN_WIDTH = 4
BOOK_WIDTH = 5
#FAN_WIDTH = 25
#LIGHT_WIDTH = 10

INCHES_TO_METERS = 0.0254
# Object detector constant 
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# colors for object detected
COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN =(0,255,0)
BLACK =(0,0,0)
# defining fonts 
FONTS = cv.FONT_HERSHEY_COMPLEX

# getting class names from classes.txt file 
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
#  setttng up opencv net

yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# object detector funciton /method
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # creating empty list to add objects data
    data_list =[]
    for (classid, box) in zip(classes, boxes):
        # define color of each, object based on its class id 
        color= COLORS[int(classid) % len(COLORS)]
        label = "%s" % (class_names[classid])
        #label = "%s : %f" % (class_names[classid], scores)

        # draw rectangle on and label on object
        cv.rectangle(image, box, color, 1)
        cv.putText(image, label, (box[0], box[1]-14), FONTS, 0.5, color, 1)
    
        # getting the data 
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        for i in range(len(class_names)):
            if classid == i:
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
        # returning list containing the object data. 
    return data_list

def focal_length_finder (measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width
    return focal_length

# distance finder function 
def distance_finder(focal_length, real_object_width, width_in_frame):
    distance = (real_object_width * focal_length) / width_in_frame
    return distance

# reading the reference image from dir 
ref_person = cv.imread('ReferenceImages/person.png')
ref_mobile = cv.imread('ReferenceImages/mobile.png')
ref_monitor = cv.imread('ReferenceImages/monitor.png')
ref_bottle = cv.imread('ReferenceImages/bottle2.png')
ref_chair = cv.imread('ReferenceImages/chair.png')
ref_laptop = cv.imread('ReferenceImages/laptop.png')
ref_keyboard = cv.imread('ReferenceImages/keyboard.png')
ref_mouse = cv.imread('ReferenceImages/mouse.png')
ref_bag = cv.imread('ReferenceImages/bag.png')
ref_pen = cv.imread('ReferenceImages/pen1.png')
ref_book = cv.imread('ReferenceImages/book.png')
ref_fan = cv.imread('ReferenceImages/fan.png')
ref_light = cv.imread('ReferenceImages/light.png')

mobile_data = object_detector(ref_mobile)
mobile_width_in_rf = mobile_data[1][1]

person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]

monitor_data = object_detector(ref_monitor)
monitor_width_in_rf = monitor_data[0][1]

bottle_data = object_detector(ref_bottle)
bottle_width_in_rf = bottle_data[0][1]

chair_data = object_detector(ref_chair)
chair_width_in_rf = chair_data[0][1]

laptop_data = object_detector(ref_laptop)
laptop_width_in_rf = laptop_data[0][1]

keyboard_data = object_detector(ref_keyboard)
keyboard_width_in_rf = keyboard_data[0][1]

mouse_data = object_detector(ref_mouse)
mouse_width_in_rf = mouse_data[0][1]

bag_data = object_detector(ref_bag)
bag_width_in_rf = bag_data[0][1]

pen_data = object_detector(ref_pen)
print("Pen Data:", pen_data)
if len(pen_data) > 0:
    pen_width_in_rf = pen_data[0][1]
else:
    raise IndexError("Not enough objects detected in the pen reference image.")

book_data = object_detector(ref_book)
book_width_in_rf = book_data[0][1]

'''#fan_data = object_detector(ref_fan)
print("Fan Data:", fan_data)
if len(fan_data) > 0:
    fan_width_in_rf = fan_data[0][1]
else:
    raise IndexError("Not enough objects detected in the fan reference image.")

#light_data = object_detector(ref_light)
print("Light Data:", light_data)
if len(light_data) > 0:
    light_width_in_rf = light_data[0][1]
else:
    raise IndexError("Not enough objects detected in the light reference image.")'''

# finding focal length 
focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)
focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)
focal_monitor = focal_length_finder(KNOWN_DISTANCE, MONITOR_WIDTH, monitor_width_in_rf)
focal_bottle = focal_length_finder(KNOWN_DISTANCE, BOTTLE_WIDTH, bottle_width_in_rf)
focal_chair = focal_length_finder(KNOWN_DISTANCE, CHAIR_WIDTH, chair_width_in_rf)
focal_laptop = focal_length_finder(KNOWN_DISTANCE, LAPTOP_WIDTH, laptop_width_in_rf)
focal_keyboard = focal_length_finder(KNOWN_DISTANCE, KEYBOARD_WIDTH, keyboard_width_in_rf)
focal_mouse = focal_length_finder(KNOWN_DISTANCE, MOUSE_WIDTH, mouse_width_in_rf)
focal_bag = focal_length_finder(KNOWN_DISTANCE, BAG_WIDTH, bag_width_in_rf)
focal_pen = focal_length_finder(KNOWN_DISTANCE, PEN_WIDTH, pen_width_in_rf)
focal_book = focal_length_finder(KNOWN_DISTANCE, BOOK_WIDTH, book_width_in_rf)
#focal_fan = focal_length_finder(KNOWN_DISTANCE, FAN_WIDTH, fan_width_in_rf)
#focal_light = focal_length_finder(KNOWN_DISTANCE, LIGHT_WIDTH, light_width_in_rf)

cap = cv.VideoCapture(1)
while True:
    ret, frame = cap.read()

    data = object_detector(frame) 
    x, y = 100,100
    for d in data:
        if d[0] =='person':
            distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
            x, y = d[2]
        elif d[0] =='cell phone':
            distance = distance_finder(focal_mobile, MOBILE_WIDTH, d[1])
            x, y = d[2]
        elif d[0] =='tvmonitor':
            distance = distance_finder(focal_monitor, MONITOR_WIDTH, d[1])
            x, y = d[2]
        elif d[0] == 'bottle':
            distance = distance_finder(focal_bottle, BOTTLE_WIDTH, d[1])
            x, y = d[2]
        elif d[0] == 'chair':
            distance = distance_finder(focal_chair, CHAIR_WIDTH, d[1])
            x, y = d[2]
        elif d[0] == 'laptop':
            distance = distance_finder(focal_laptop, LAPTOP_WIDTH, d[1])
            x, y = d[2]
        elif d[0] == 'keyboard':
            distance = distance_finder(focal_keyboard, KEYBOARD_WIDTH, d[1])
            x, y = d[2]
        elif d[0] == 'mouse':
            distance = distance_finder(focal_mouse, MOUSE_WIDTH, d[1])
            x, y = d[2]
        elif d[0] == 'backpack': 
            distance = distance_finder(focal_bag, BAG_WIDTH, d[1])
            x, y = d[2]
        elif d[0] == 'pen':
            distance = distance_finder(focal_pen, PEN_WIDTH, d[1])
            x, y = d[2]
        elif d[0] == 'book':
            distance = distance_finder(focal_book, BOOK_WIDTH, d[1])
            x, y = d[2]
        
        
        # Convert distance to meters
        distance_in_meters = distance * INCHES_TO_METERS
        cv.rectangle(frame, (x, y-3), (x + 150, y + 23),BLACK, -1 )
        cv.putText(frame, f'Dis: {round(distance_in_meters,2)} meters', (x+5,y+13), FONTS, 0.48, GREEN, 1)

    cv.imshow('frame',frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break
cv.destroyAllWindows()
cap.release()

 