import colorsys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import cv2
import time
import sys

import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import image_preporcess

class YOLO(object):
    _defaults = {
        "model_path": 'logs/000/trained_weights_final.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": '4_CLASS_test_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "text_size" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = image_preporcess(np.copy(image), tuple(reversed(self.model_image_size)))
            image_data = boxed_image

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.shape[0], image.shape[1]],#[image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        #print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        thickness = (image.shape[0] + image.shape[1]) // ((image.shape[0] + image.shape[1])*2)
        fontScale=1
        ObjectsList = []
        
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            scores = '{:.2f}'.format(score)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.shape[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.shape[1], np.floor(right + 0.5).astype('int32'))

            mid_h = (bottom-top)/2+top
            mid_v = (right-left)/2+left

            # put object rectangle
            cv2.rectangle(image, (left, top), (right, bottom), self.colors[c], thickness)

            # get text size
            (test_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, thickness/self.text_size, 1)

            # put text rectangle
            cv2.rectangle(image, (left, top), (left + test_width, top - text_height - baseline), self.colors[c], thickness=cv2.FILLED)

            # put text above rectangle
            cv2.putText(image, label, (left, top-2), cv2.FONT_HERSHEY_SIMPLEX, thickness/self.text_size, (0, 0, 0), 1)

            # add everything to list
            ObjectsList.append([top, left, bottom, right, mid_v, mid_h, label, scores])

        return image, ObjectsList

    def close_session(self):
        self.sess.close()

    def detect_img(self, image):

        r_image, ObjectsList = self.detect_image(image)
        return r_image, ObjectsList


def drawLines(frame):
    # Display Lines and Regions
    cv2.line(frame, (frame.shape[1]//3, 0), (frame.shape[1]//3, frame.shape[0]), (255, 0, 0), 1, 1)
    cv2.line(frame, (2*(frame.shape[1]//3), 0), (2*(frame.shape[1]//3), frame.shape[0]), (255, 0, 0), 1, 1)
    cv2.line(frame, (0, frame.shape[0]//3), (frame.shape[1], frame.shape[0]//3), (255, 0, 0), 1, 1)
    cv2.line(frame, (0, 2*(frame.shape[0]//3)), (frame.shape[1], 2*(frame.shape[0]//3)), (255, 0, 0), 1, 1)

    cv2.putText(frame, "Region 1", (frame.shape[1]//3 - 70, frame.shape[0]//3 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(frame, "Region 2", (2*(frame.shape[1]//3) - 70, frame.shape[0]//3 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(frame, "Region 3", (frame.shape[1] - 70, frame.shape[0]//3 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(frame, "Region 4", (frame.shape[1]//3 - 70, 2*(frame.shape[0]//3) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(frame, "Region 5", (frame.shape[1] - 70, 2*(frame.shape[0]//3) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(frame, "Region 6", (frame.shape[1]//3 - 70, frame.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(frame, "Region 7", (2*(frame.shape[1]//3) - 70, frame.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(frame, "Region 8", (frame.shape[1] - 70, frame.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

def append_log(log_list, second, region_number, efsb_counter):

    # if situation_code == 1:
    #     generated_log = "A Worm is in region no. "+str(region_number)+"."
    # elif situation_code == 2:
    #     generated_log = "Eggplant in region no. "+str(region_number)+" has an EFSB."

    generated_log = "Region["+str(region_number)+"] update with "+str(efsb_counter)+" found!"

    print(generated_log)
    log_list.append((second,generated_log))

    return 0

def detectRegion(frame_width, frame_height, object_centroid):

    cX = object_centroid[0]
    cY = object_centroid[1]

    temp_coord = [1,1]

    temp_coord[0] = (0 if (cX < frame_width/3) else
                    1 if (cX >= frame_width/3) and (cX < (2*frame_width)/3) else
                    2)

    temp_coord[1] = (0 if (cY < frame_height/3) else
                    1 if (cY >= frame_height/3) and (cY < (2*frame_height)/3) else
                    2)

    return( 1 if temp_coord == [0,0] else
            4 if temp_coord == [1,0] else
            3 if temp_coord == [2,0] else
            2 if temp_coord == [0,1] else
            0 if temp_coord == [1,1] else
            7 if temp_coord == [2,1] else
            6 if temp_coord == [0,2] else
            5 if temp_coord == [1,2] else 
            8 ) 


import math  

def calculateDistance(a,b):  
    # computes distance between 2 points 
     
     dist = math.sqrt((a[1] - b[1])**2 + (a[0] - b[0])**2)  
     return dist  

def swapxy(a):

    return (a[1],a[0])

def coordinate_in_box(coordinate, box):

    cx = coordinate[0]
    cy = coordinate[1]

    if box[1] < cy < box[3]:
        if box[0] < cx < box[2]:
            return True

    return False

if __name__=="__main__":
    yolo = YOLO()

    # set start time to current time
    start_time = time.time()
    # displays the frame rate every second
    display_time = 1
    # Set primarry FPS to 0
    fps = 0

    log = []

    # frame skipping mechanics
    current_frame = 0
    frame_skip = 30
    video_fps = 15
    second = 0

    # initializing region states
    region_states = [0 for i in range(0,9)]

    # video output

    # we create the video capture object cap
    cap = cv2.VideoCapture(sys.argv[1])
    if not cap.isOpened():
        raise IOError("We cannot open webcam")

    fgbg = cv2.createBackgroundSubtractorMOG2()
    
    scale = 0.5
    
    ret, frame = cap.read() #initial read
    frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    

    (h, w) = frame.shape[:2]
    out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*"XVID"), 15, (w,h), True)

    black = np.zeros(frame.shape, dtype=np.uint8)
    black2 = np.zeros(frame.shape, dtype=np.uint8)
    prev_efsb_coords = []
    efsb_coords = []

    while True:

        ret, frame = cap.read()

        current_frame += 1

        if current_frame%frame_skip != 0:
            continue

        second = current_frame//video_fps

        # initializing region states
        new_region_states = [0 for i in range(0,9)]

        # resize our captured frame if we need

        if frame is None:
            break

        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # detect object on our frame
        r_image, ObjectsList = yolo.detect_img(frame)

        # get center of each object that is an efsb
        sample_string = ''
        # efsb_count = 0
        prev_efsb_coords = efsb_coords
        efsb_coords = []
        eggplant_boxes = []
        for index, obj in enumerate(ObjectsList):

            str_obj = [str(i) for i in obj]
            concat = sample_string.join(str_obj)

            if 'efsb' in concat:
                x = int((obj[0]+obj[2])/2)
                y = int((obj[1]+obj[3])/2)
                efsb_coordinate = (x,y)
                black[x,y] = [0,255,0]
                efsb_coords.append(efsb_coordinate)

                region_number = detectRegion(frame.shape[0], frame.shape[1], efsb_coordinate)

                new_region_states[region_number] += 1

            elif 'eggplant' in concat:
                cv2.rectangle(black, (obj[1], obj[0]), (obj[3], obj[2]), [0,255,0], 1)
                cv2.rectangle(black2, (obj[1], obj[0]), (obj[3], obj[2]), [0,255,0], 1)
                eggplant_boxes.append((obj[0:4]))

        # print(efsb_coords)
        # print(prev_efsb_coords)

        if prev_efsb_coords != [] and efsb_coords != []:
            for a in prev_efsb_coords:
                
                smallest_distance = 9999
                smallest_index = 0

                for bindex, b in enumerate(efsb_coords):
                    # print(a, b, calculateDistance(a,b))
                    k = calculateDistance(a,b)
                    if smallest_distance > k:
                        smallest_distance = k
                        smallest_index = bindex

                if smallest_distance < 20:
                    cv2.line(black2, swapxy(efsb_coords[smallest_index]), swapxy(a), (0,255,0), 1)

        same = True
        for index, element in enumerate(new_region_states):
            if element != region_states[index]:
                append_log(log, second, index, element)
                same = False

        if not same:
            region_states = new_region_states

        # for coordinate in efsb_coords:

        #     recorded = False
        #     region_number = detectRegion(frame.shape[0], frame.shape[1], coordinate)
            
        #     for box in eggplant_boxes:

        #         if coordinate_in_box(coordinate, box):
        #             append_log(log, second, 2, region_number)
        #             recorded = True
        #             break
        #         # else:
        #     if not recorded:
        #         append_log(log, second, 1, region_number)
                    # break

        # add lines to our image
        drawLines(r_image)

        # show us frame with detection
        # cv2.imshow("input", r_image)
        # cv2.imshow("black2", black2)
        out.write(r_image)

        # cv2.imshow("mog", fgmask)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

        # calculate FPS
        fps += 1
        TIME = time.time() - start_time
        if TIME > display_time:
            print("FPS:", fps / TIME)
            fps = 0 
            start_time = time.time()

    f = open("log_latest_run.txt","w")
    string_to_write = ""
    for line in log:
        string_to_write = string_to_write + "Second #{} | {} \n".format(line[0], line[1])
    f.write(string_to_write)
    
    cv2.imwrite("EFSB_trajectory.jpg",black)        
    cv2.imwrite("EFSB_line_trajectory.jpg",black2)        
    out.release()
    cap.release()
    cv2.destroyAllWindows()
    yolo.close_session()
