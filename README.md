# Yolov4-Detector-and-Distance-Estimator

*Find the distance from the object to the camera using the YoloV4 object detector, here we will be using a single camera :camera:, detailed explanation of distance estimation is available in another repository* [**Face detection and Distance Estimation using single camera**]

- Here we are targeting some classes only.

- implementation detail available on [_**Darknet**_](https://github.com/pjreddie/darknet)

---

# TO DO

- [x] Finding the distance of multiple objects at the same time.

## Installation you need opencv-contrib-python

[opencv contrib](https://pypi.org/project/opencv-contrib-python/)

### **windows**

```pip
pip install opencv-contrib-python==4.5.3.56
```

### **Linux or Mac**

```pip
pip3 install opencv-contrib-python==4.5.3.56
```

then just clone this repository and you are good to go.

I have used tiny weights, check out more on darknet GitHub for more

---

## Add more Classes(Objects) for Distance Estimation

You will make changes on these particular lines [***DistanceEstimation.py***]

if classid ==0: # person class id 
    data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
elif classid ==67: # cell phone
    data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
    
# Adding more classes for distance estimation 

elif classid ==2: # car
    data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])

elif classid ==15: # cat
    data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])

# In that way you can include as many classes as you want 

    # returning list containing the object data. 
return data_list

# Reading images and getting focal length

You have to make changes on these lines 📝 [***DistanceEstimation.py***])
there are two situations, if the object(classes) in the single image then, here you can see my reference image <img src='ReferenceImages/image4.png' width=200> 

# reading reference images 
ref_person = cv.imread('ReferenceImages/person.png')
ref_mobile = cv.imread('ReferenceImages/mobile.png')

# calling the object detector function to get the width or height of the object
# getting pixel width for person
person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]

# Getting pixel width for cell phone
mobile_data = object_detector(ref_mobile)
mobile_width_in_rf = mobile_data[1][1]

# getting pixel width for cat
cat_data = object_detector(ref_person)
cat_width_in_rf = person_data[2][1]

# Getting pixel width for car
car_data = object_detector(ref_person)
car_width_in_rf = person_data[3][1]

# if there is single class(object) in reference image then you approach it that way 👍
# reading the reference image from dir 
ref_person = cv.imread('ReferenceImages/person_ref_img.png')
ref_car = cv.imread('ReferenceImages/car_ref_img.png.png')
ref_cat = cv.imread('ReferenceImages/cat_ref_img.png')
ref_mobile = cv.imread('ReferenceImages/ref_cell_phone.png')

# Checking object detection on the reference image 
# getting pixel width for person
person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]

# Getting pixel width for cell phone
mobile_data = object_detector(ref_mobile)
mobile_width_in_rf = mobile_data[0][1]

# getting pixel width for cat
cat_data = object_detector(ref_cat)
cat_width_in_rf = person_data[0][1]

# Getting pixel width for car
car_data = object_detector(ref_car)
car_width_in_rf = person_data[0][1]

# Then you find the Focal length for each
def focal_length_finder (measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width
    return focal_length

# Then you find the distance for each  
def distance_finder(focal_length, real_object_width, width_in_frame):
    distance = (real_object_width * focal_length) / width_in_frame
    return distance



 
   


 

