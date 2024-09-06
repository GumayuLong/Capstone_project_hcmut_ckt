import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf
import pandas as pd
import matplotlib as plt
import math as m

# ********************************************** Global parameters **************************************************************

X = []
y = []

epsilon_push = []
epsilon_pushbh = []
epsilon_topspin = []
epsilon_topspinbh = []

black = (0, 0, 0)
green = (0, 255, 0)
red = (50, 50, 255)

label = "Warmup...."
n_time_steps = 4
lm_list = []
lm_list_data = []
arr_lm_list_data = []

eva_push = []
eva_topspin = []
eva_pushbh = []
eva_topspinbh = []

data_push = []
data_topspin = []
data_topspinbh = []
data_pushbh = []

positionPose2 = []

wrong_push = []
wrong_pushbh = []
wrong_topspin = []
wrong_topspinbh = []

list_pose_push = []
list_pose_topspin = []
list_pose_topspinbh = []
list_pose_pushbh = []

lm_list_data_topspin = []
lm_list_data_topspinbh = []
lm_list_data_push = []
lm_list_data_pushbh = []

# ************************************************************************************************

# Read file PUSH.txt
push_df = pd.read_csv("PUSH.txt")
dataset_push = push_df.iloc[:,1:].values
n_sample = len(dataset_push)
for i in range(n_time_steps, n_sample):
    X.append(dataset_push[i-n_time_steps:i,:])

arr_flat_push = [item for sublist in dataset_push for item in sublist]
data_push.append(arr_flat_push)
my_array_push = np.array(data_push).flatten()

# ************************************************************************************************

# Read file TOPSPIN.txt
topspin_df = pd.read_csv("TOPSPIN.txt")
dataset_topspin = topspin_df.iloc[:,1:].values
n_sample = len(dataset_topspin)
for i in range(n_time_steps, n_sample):
    X.append(dataset_topspin[i-n_time_steps:i,:])

arr_flat_topspin = [item for sublist in dataset_topspin for item in sublist]
data_topspin.append(arr_flat_topspin)
my_array_topspin = np.array(data_topspin).flatten()

# ************************************************************************************************

# Read file TOPSPINBACKHAND.txt
topspinbh_df = pd.read_csv("TOPSPINBACKHAND.txt")
dataset_topspinbh = topspinbh_df.iloc[:,1:].values
n_sample = len(dataset_topspinbh)
for i in range(n_time_steps, n_sample):
    X.append(dataset_topspinbh[i-n_time_steps:i,:])

arr_flat_topspinbh = [item for sublist in dataset_topspinbh for item in sublist]
data_topspinbh.append(arr_flat_topspinbh)
my_array_topspinbh = np.array(data_topspinbh).flatten()

# ************************************************************************************************

# Read file PUSHBACKHAND.txt
pushbh_df = pd.read_csv("PUSHBACKHAND.txt")
dataset_pushbh = pushbh_df.iloc[:,1:].values
n_sample = len(dataset_pushbh)
for i in range(n_time_steps, n_sample):
    X.append(dataset_pushbh[i-n_time_steps:i,:])

arr_flat_pushbh = [item for sublist in dataset_pushbh for item in sublist]
data_pushbh.append(arr_flat_pushbh)
my_array_pushbh = np.array(data_pushbh).flatten()

# ************************************************************************************************
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

model = tf.keras.models.load_model("model.h5")

# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/success4-30fps-front.mov")
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/fail.mov")
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/test3.mp4")
cap = cv2.VideoCapture(0)

# ****************************************** PUSH ************************************************
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/test6.mp4")
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/push2.mp4")
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/test9.mp4")
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/test10.mp4")

# ************************************************************************************************
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/test5.mp4")
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/dataset_Push.mp4")
# ************************************************************************************************
# ************************************************************************************************

# TOPSPIN
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/test7.mp4")
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/test4.mp4")
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/dataset_Topspin4.mp4")

# ************************************************************************************************
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/Mistake_topspin.mp4")
# ************************************************************************************************
# ************************************************************************************************

# ****************************************** PUSH BACKHAND ******************************************
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/pushbackhand.mp4")
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/pushbackhand3.mp4")

# ****************************************** DATASET ******************************************
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/dataset_Pushbackhand.mp4")
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/push3.mp4")
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/pushbackhand2.mp4")
# **************************************************************************************************
# ********************************* Correct EVALUATE PUSH BACKHAND *********************************
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/wrongPushBHTest.mp4") 
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/mistake_3.mp4") 
# **************************************************************************************************

# ****************************************** TOPSPIN BACKHAND ******************************************
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/topspinbackhand2.mp4")

# ****************************************** GOOD TESTCASE ******************************************
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/testTopspinbh2.mp4")
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/testTopspinbh4.mp4")

# ****************************************** DATASET ******************************************
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/testTopspinbackhand.mp4")

# ********************************************* BEST CASE *********************************************
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/dataset_Topspinbackhand3.mp4")
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/dataset_Topspinbackhand.mp4")
# ********************************************* Correct EVALUATE PUSH BACKHAND *********************************************
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/mistake_topspinbh_1.mp4") 

# ****************************************************************************************************
# ********************************************* MISTAKES *********************************************
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/mistake_topspinbh_2.mp4") 

# **************************************** GOOD TESTCASE WRONG POSE (PUSH BACKHAND) ***************************************
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/wrongPushBHTest.mp4") 
# *************************************************************************************************************************
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/mistake_3.mp4") 
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/mistake_4.mp4") 
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/mistake_5.mp4") 
# ****************************************************************************************************

def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

# =================================================== Hàm tìm khoảng cách giữa 2 điểm pose ===================================================
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2-x1)**2+(y2-y1)**2)
    return dist

# =================================================== Hàm tính góc bao nhiêu độ ===================================================
def findAngle(p1, p2, p3):
    x1 = p1.x
    y1 = p1.y
    x2 = p2.x
    y2 = p2.y
    x3 = p3.x
    y3 = p3.y
    angle = m.degrees(m.atan2(y3 - y2, x3 - x2) -
                             m.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 180
    # theta = m.acos( (y2 -y1)*(-y1) / (m.sqrt(
    #     (x2 - x1)**2 + (y2 - y1)**2 ) * y1) )
    # degree = int(180/m.pi)*theta
    return angle

def Calc_Angle(shoulder, elbow, wrist):
   # Extract coordinates from landmarks
    shoulder = np.array([shoulder.x, shoulder.y, shoulder.z])
    elbow = np.array([elbow.x, elbow.y, elbow.z])
    wrist = np.array([wrist.x, wrist.y, wrist.z])
    
    # Vectors representing the arm segments
    upper_arm = elbow - shoulder
    forearm = wrist - elbow
    
    # Normalize the vectors
    upper_arm_unit = upper_arm / np.linalg.norm(upper_arm)
    forearm_unit = forearm / np.linalg.norm(forearm)
    
    # Calculate dot product and angle between vectors
    dot_product = np.dot(upper_arm_unit, forearm_unit)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    
    # Convert angle to degrees
    angle_degrees = np.degrees(angle)
    
    return angle_degrees

def findAngleShoulder(p1, p2, p3):
    x1 = p1.x
    y1 = p1.y
    x2 = p2.x
    y2 = p2.y
    x3 = p3.x
    y3 = p3.y
    angle = m.degrees(m.atan2(y3 - y2, x3 - x2) -
                             m.atan2(y1 - y2, x1 - x2))
    # if angle < 0:
    #     angle += 180
    # theta = m.acos( (y2 -y1)*(-y1) / (m.sqrt(
    #     (x2 - x1)**2 + (y2 - y1)**2 ) * y1) )
    # degree = int(180/m.pi)*theta
    return angle

def findAngleElbowPush(p1, p2, p3):
    x1 = p1.x
    y1 = p1.y
    x2 = p2.x
    y2 = p2.y
    x3 = p3.x
    y3 = p3.y
    angle = m.degrees(m.atan2(y3 - y2, x3 - x2) -
                             m.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    # theta = m.acos( (y2 -y1)*(-y1) / (m.sqrt(
    #     (x2 - x1)**2 + (y2 - y1)**2 ) * y1) )
    # degree = int(180/m.pi)*theta
    return angle

# =================================================== Hàm tính điểm trung bình giữa 2 điểm pose 
def find_midpoint_landmark(landmark1, landmark2):
    return (landmark1.x + landmark2.x) / 2, (landmark1.y + landmark2.y) / 2, (landmark1.z + landmark2.z) / 2

# =================================================== Hàm dời tất cả các landmark sang điểm trung tâm giữa hai vai 
def shift_landmarks_to_midpoint(landmarks, midpoint):
    for landmark in landmarks:
        landmark.x -= midpoint[0]
        landmark.y -= midpoint[1]
        landmark.z -= midpoint[2]

# =================================================== Hàm vẽ các điểm pose và xuất ra video 
def draw_landmark_on_image(mpDraw, results, img):
    # mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    # for id, lm in enumerate(results.pose_landmarks.landmark):
    #     h, w, c = img.shape
    # print(lm)
    #     cx, cy = int(lm.x * w), int(lm.y * h)
    #     cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    for id, lm in enumerate(results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_SHOULDER.value:mpPose.PoseLandmark.LEFT_PINKY.value]):
        h, w, c = img.shape
        # print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 3, (0, 0, 255), cv2.FILLED)

    upper_body_connections = [
            (mpPose.PoseLandmark.LEFT_SHOULDER.value, mpPose.PoseLandmark.LEFT_ELBOW.value),
            (mpPose.PoseLandmark.RIGHT_SHOULDER.value, mpPose.PoseLandmark.RIGHT_ELBOW.value),
            (mpPose.PoseLandmark.LEFT_ELBOW.value, mpPose.PoseLandmark.LEFT_WRIST.value),
            (mpPose.PoseLandmark.RIGHT_ELBOW.value, mpPose.PoseLandmark.RIGHT_WRIST.value),
            (mpPose.PoseLandmark.LEFT_SHOULDER.value, mpPose.PoseLandmark.RIGHT_SHOULDER.value),
        ]
    for connection in upper_body_connections:
            start_idx, end_idx = connection
            start_point = (int(upper_body_landmarks[start_idx].x * w), int(upper_body_landmarks[start_idx].y * h))
            end_point = (int(upper_body_landmarks[end_idx].x * w), int(upper_body_landmarks[end_idx].y * h))

            # Draw a line between the start and end points
            cv2.line(img, start_point, end_point, (0, 255, 0), 2, cv2.FILLED)  # Green line for each connection
    return img

def draw_landmark_on_image_elbow_error(mpDraw, results, img):
    # mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    # for id, lm in enumerate(results.pose_landmarks.landmark):
    #     h, w, c = img.shape
        # print(id, lm)
    #     cx, cy = int(lm.x * w), int(lm.y * h)
    #     cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    for id, lm in enumerate(results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_SHOULDER.value:mpPose.PoseLandmark.LEFT_PINKY.value]):
        h, w, c = img.shape
        # print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 3, (0, 0, 255), cv2.FILLED)

    upper_body_connections = [
            (mpPose.PoseLandmark.LEFT_SHOULDER.value, mpPose.PoseLandmark.LEFT_ELBOW.value),
            (mpPose.PoseLandmark.RIGHT_SHOULDER.value, mpPose.PoseLandmark.RIGHT_ELBOW.value),
            (mpPose.PoseLandmark.LEFT_ELBOW.value, mpPose.PoseLandmark.LEFT_WRIST.value),
            (mpPose.PoseLandmark.RIGHT_ELBOW.value, mpPose.PoseLandmark.RIGHT_WRIST.value),
            (mpPose.PoseLandmark.LEFT_SHOULDER.value, mpPose.PoseLandmark.RIGHT_SHOULDER.value),
        ]
    for connection in upper_body_connections:
            start_idx, end_idx = connection
            start_point = (int(upper_body_landmarks[start_idx].x * w), int(upper_body_landmarks[start_idx].y * h))
            end_point = (int(upper_body_landmarks[end_idx].x * w), int(upper_body_landmarks[end_idx].y * h))

            # Draw a line between the start and end points
            # if start_idx == mpPose.PoseLandmark.RIGHT_SHOULDER.value and end_idx == mpPose.PoseLandmark.RIGHT_ELBOW.value:
            #     cv2.line(img, start_point, end_point, (50, 50, 255), 2, cv2.FILLED)  # Green line for each connection
            if start_idx == mpPose.PoseLandmark.RIGHT_ELBOW.value and end_idx == mpPose.PoseLandmark.RIGHT_WRIST.value:
                cv2.line(img, start_point, end_point, (50, 50, 255), 2, cv2.FILLED)
            else:
                cv2.line(img, start_point, end_point, (0, 255, 0), 2, cv2.FILLED)
    return img

def draw_landmark_on_image_shoulder_error(mpDraw, results, img):
    # mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    # for id, lm in enumerate(results.pose_landmarks.landmark):
    #     h, w, c = img.shape
        # print(id, lm)
    #     cx, cy = int(lm.x * w), int(lm.y * h)
    #     cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    for id, lm in enumerate(results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_SHOULDER.value:mpPose.PoseLandmark.LEFT_PINKY.value]):
        h, w, c = img.shape
        # print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 3, (0, 0, 255), cv2.FILLED)

    upper_body_connections = [
            (mpPose.PoseLandmark.LEFT_SHOULDER.value, mpPose.PoseLandmark.LEFT_ELBOW.value),
            (mpPose.PoseLandmark.RIGHT_SHOULDER.value, mpPose.PoseLandmark.RIGHT_ELBOW.value),
            (mpPose.PoseLandmark.LEFT_ELBOW.value, mpPose.PoseLandmark.LEFT_WRIST.value),
            (mpPose.PoseLandmark.RIGHT_ELBOW.value, mpPose.PoseLandmark.RIGHT_WRIST.value),
            (mpPose.PoseLandmark.LEFT_SHOULDER.value, mpPose.PoseLandmark.RIGHT_SHOULDER.value),
            ]
    for connection in upper_body_connections:
            start_idx, end_idx = connection
            start_point = (int(upper_body_landmarks[start_idx].x * w), int(upper_body_landmarks[start_idx].y * h))
            end_point = (int(upper_body_landmarks[end_idx].x * w), int(upper_body_landmarks[end_idx].y * h))

            # Draw a line between the start and end points
            if start_idx == mpPose.PoseLandmark.RIGHT_SHOULDER.value and end_idx == mpPose.PoseLandmark.RIGHT_ELBOW.value:
                cv2.line(img, start_point, end_point, (50, 50, 255), 2, cv2.FILLED)  # Green line for each connection
            # if start_idx == mpPose.PoseLandmark.RIGHT_ELBOW.value and end_idx == mpPose.PoseLandmark.RIGHT_WRIST.value:
            #     cv2.line(img, start_point, end_point, (50, 50, 255), 2, cv2.FILLED)
            else:
                cv2.line(img, start_point, end_point, (0, 255, 0), 2, cv2.FILLED)
    return img

def draw_class_on_image(label, img, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = color
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img

# Điều kiện của thế đánh TOPSPIN FOREHAND (???) ======= ()
# def topspinCondition():
#     return 0

# Điều kiện của thế đánh PUSH FOREHAND (Góc khuỷu tay khi bắt đầu thường khoảng 90 độ hoặc nhỏ hơn xíu (tùy người) và góc khuỷu tay khi ra đòn trong khoảng từ 135 -> 150 độ) ======= (DONE)
def pushCondition(checkAngleElbow, mpDraw, result, imgInput, label):
    if angleCheck(checkAngleElbow, 70, 150, 20):
        # print(checkAngleElbow)
        img = draw_landmark_on_image(mpDraw, result, imgInput)
        img = draw_class_on_image(label, imgInput, green)
        return img
    else:
        # print(checkAngleElbow)
        # img = draw_class_on_image_error(imgInput)
        img = draw_class_on_image(label, imgInput, red)
        img_error = draw_landmark_on_image_elbow_error(mpDraw, result, imgInput)
        return img_error

# Điều kiện của thế đánh PUSH BACKHAND (góc khuỷu tay không duỗi thẳng 180 độ và gập tay không dưới 90 độ) ======= (DONE)
def pushBackHandCondition(checkAngleElbow, mpDraw, result, imgInput, label):
    if angleCheck(checkAngleElbow, 90, 180, 10):
        # print(checkAngleElbow)
        img = draw_class_on_image(label, imgInput, green)
        img = draw_landmark_on_image(mpDraw, result, imgInput)
        return img
    else :
        print("PUSH BACKHAND Wrong!")
        # print(checkAngleElbow)
        # img = draw_class_on_image_error (imgInput)
        img = draw_class_on_image (label, imgInput, red)
        img_error = draw_landmark_on_image_elbow_error(mpDraw, result, imgInput)
        return img_error

# Điều kiện của thế đánh TOPSPIN BACKHAND (Angle of shoulder <= 90 degree and angle of elbow between 90 -> 150 degree) ======= (DONE)
def topspinBackHandCondition(checkAngleShoulder, checkAngleElbow, mpDraw, result, imgInput, label):
    if angleCheck(checkAngleElbow, 90, 150, 20) and angleCheck(checkAngleShoulder, 0, 110, 10):
        # print(checkAngleShoulder)
        img = draw_class_on_image(label, imgInput, green)
        img = draw_landmark_on_image(mpDraw, result, imgInput)
        return img
    elif (90 - checkAngleElbow) >= 150:
        print("ELBOW - TOPSPIN BACKHAND Wrong!")
        # print(checkAngleElbow)
        img = draw_class_on_image (label, imgInput, red)
        img_error = draw_landmark_on_image_elbow_error(mpDraw, result, imgInput)
        return img_error
    elif checkAngleShoulder >= 110:
        # print("SHOULDER Wrong!")
        # print(checkAngleShoulder)
        img = draw_class_on_image (label, imgInput, red)
        img_error = draw_landmark_on_image_shoulder_error(mpDraw, result, imgInput)
        return img_error

# Hàm kiểm tra góc
def angleCheck(myAngle, minAngle, maxAngle, offset):
        return minAngle - offset < myAngle < maxAngle

def evaluatePose(A, B, eva):
    for i in range(len(A)):
        for j in range(len(B)):
            C = B[j]
            for z in range(len(C)):
                if(z == i):
                    result = abs(abs(A[i]) - abs(C[z]))
                    eva.append(result)
                    break
    return eva

def biasDataset(dataset, evaluatePose, epsilon):
    total = 0
    check = 0
    resultBias = 0
    for i in range(len(evaluatePose)):
        for j in range(len(evaluatePose)):
            if(i==j and i%3 == 0):
                result_1 = m.sqrt(evaluatePose[j]*evaluatePose[j] + evaluatePose[j+1]*evaluatePose[j+1] + evaluatePose[j+2]*evaluatePose[j+2])
                result = m.sqrt(dataset[i]*dataset[i] + dataset[i+1]*dataset[i+1] + dataset[i+2]*dataset[i+2])
                # bias = result_1/result
                check += result*result
                bias = result_1/(result)
                # bias = result_1/distanceShoulder
                epsilon.append(bias)
                total += result_1*result_1
                resultBias += bias*bias
                break
    print("--Total error--: " + str(m.sqrt(resultBias)))
    # print("Score: " + str(100 - m.sqrt(resultBias)))
    if (100 - m.sqrt(resultBias) < 50):
        print("--Evaluating: Bad!--")
    elif (100 - m.sqrt(resultBias) >= 50 and 1 - m.sqrt(resultBias) < 70):
        print("--Evaluating: Good!--")
    elif (100 - m.sqrt(resultBias) >= 70):
        print("--Evaluating: Excellent!--")
    return total

def findWrongPose(eva, wrong):
    for i in range(len(eva)):
        if(eva[i] >= 0.25):
            wrong.append(eva[i])
    return wrong

def sumWrongPose(wrong):
    sumWrong = 0
    for i in range(len(wrong)):
        # if(wrong[i] >= 0.2):
        sumWrong += wrong[i]
    return sumWrong

def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    results = model.predict(lm_list)

    # # Tọa độ 2 khuỷu tay
    # left_elbow = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_ELBOW.value]
    # right_elbow = results.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_ELBOW.value]

    # # Tọa độ 2 cổ tay
    # left_wrist = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_WRIST.value]
    # right_wrist = results.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_WRIST.value]
        
    print(results[0][0], results[0][1], results[0][2], results[0][3])

    if results[0][0] > 0.9:
        label = "TOPSPIN"
    elif results[0][1] > 0.9:
        label = "TOPSPIN BACKHAND"
    elif results[0][2] > 0.9:
        label = "PUSH"
    elif results[0][3] > 0.9:
        label = "PUSH BACKHAND"
    else:
        label = "UNDEFINE"
    return label

i = 0
warmup_frames = 4

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    results2 = pose.process(imgRGB)
    i = i + 1
    if i > warmup_frames:
        # print("Start detect....")
        if results.pose_landmarks:

            # Finding coordinate of shoulders
            left_shoulder = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = results.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_SHOULDER.value]

            # Finding coordinate of elbows
            left_elbow = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_ELBOW.value]
            right_elbow = results.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_ELBOW.value]

            # Finding coordinate of wrists
            left_wrist = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = results.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_WRIST.value]
            
            # Tính toán điểm trung điểm giữa hai vai
            midpoint = find_midpoint_landmark(left_shoulder, right_shoulder)
            
            # Dời tất cả các landmark về gốc tọa độ mới là điểm trung điểm giữa hai vai
            shift_landmarks_to_midpoint(results.pose_landmarks.landmark, midpoint)
            c_lm = make_landmark_timestep(results)
            upper_body_landmarks = results2.pose_landmarks.landmark[:mpPose.PoseLandmark.LEFT_HIP.value]
            # upper_body_landmarks = results.pose_landmarks.landmark[:mpPose.PoseLandmark.LEFT_HIP.value]

            lm_list.append(c_lm)
            if len(lm_list) == n_time_steps:
                # predict
                t1 = threading.Thread(target=detect, args=(model, lm_list,))
                t1.start()
                if(label == "PUSH"):
                    # pushCondition(checkAngleElbowPush, mpDraw, results2, img, label)
                    list_pose_push.append(lm_list)
                    arr_flat_lm_list_push = [item for sublist in list_pose_push for item in sublist]
                    lm_list_data_push = np.array(arr_flat_lm_list_push)
                elif(label == "TOPSPIN"):
                    list_pose_topspin.append(lm_list)
                    arr_flat_lm_list_topspin = [item for sublist in list_pose_topspin for item in sublist]
                    lm_list_data_topspin = np.array(arr_flat_lm_list_topspin)
                elif(label == "TOPSPIN BACKHAND"):
                    # topspinBackHandCondition(checkAngleShoulder, checkAngleElbow, mpDraw, results2, img, label)
                    list_pose_topspinbh.append(lm_list)
                    arr_flat_lm_list_topspinbh = [item for sublist in list_pose_topspinbh for item in sublist]
                    lm_list_data_topspinbh = np.array(arr_flat_lm_list_topspinbh)
                elif(label == "PUSH BACKHAND"):
                    # pushBackHandCondition(checkAngleElbow, mpDraw, results2, img, label)
                    list_pose_pushbh.append(lm_list)
                    arr_flat_lm_list_pushbh = [item for sublist in list_pose_pushbh for item in sublist]
                    lm_list_data_pushbh = np.array(arr_flat_lm_list_pushbh)
                lm_list = []
            # checkDistance = findDistance(left_wrist.x, right_wrist.x, left_shoulder.x, right_shoulder.x)
            # Tính góc khuỷu tay =============================================================================
            # checkAngleElbow = findAngle(right_shoulder, right_elbow, right_wrist)
            # checkAngleShoulder = 360 - findAngle(left_shoulder, right_shoulder, right_elbow) - findAngle(right_shoulder, right_elbow, right_wrist)
            # Tính góc vai ===================================================================================
            # checkAngleShoulder = 180 - findAngleShoulder(left_shoulder, right_shoulder, right_elbow)
            # checkAngleShoulder = Calc_Angle(right_elbow, right_shoulder, left_shoulder)
            # checkAngleElbowPush = findAngleElbowPush(right_wrist, right_elbow, right_shoulder)
            # checkAngleElbowPush = Calc_Angle(right_wrist, right_elbow, right_shoulder)

# ================================================================= PUSH BACKHAND CONDITION =================================================================
            # if label == "PUSH BACKHAND":
                # pushBackHandCondition(checkAngleElbow, mpDraw, results2, img, label)
            # if label == "PUSH":
                # pushCondition(checkAngleElbowPush, mpDraw, results2, img, label)
                # print(dataset)
                # flattened_list = sum(lm_list, [])
                # positionPose.append(flattened_list)
                # evaluatePose(arr_flat, flattened_list, eva)
            # elif label == "TOPSPIN BACKHAND":
                # topspinBackHandCondition(checkAngleShoulder, checkAngleElbow, mpDraw, results2, img, label) # Góc ở khuỷu tay với vai phải bé hơn 90 độ và góc ở elbow từ 90 -> 150 độ
            # else: 
            img = draw_class_on_image(label, img, green)
            
            positionPose2 = np.array(lm_list).flatten()
            img = draw_landmark_on_image(mpDraw, results2, img)
            # Vẽ lại pose trên window xuất ra màn hình
            # img = draw_landmark_on_image(mpDraw, results2, img)

    # img = draw_class_on_image(label, img, green)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

list_lm_list_data = [lm_list_data_topspin, lm_list_data_topspinbh, lm_list_data_pushbh, lm_list_data_push]
arr_lm_list_data = np.array(list_lm_list_data, dtype = object)
# print(arr_lm_list_data)
for i in arr_lm_list_data: 
    if(len(i) != 0):
        if(len(i) == len(lm_list_data_push)):
            print("PUSH")
            evaluatePose(my_array_push, lm_list_data_push, eva_push)
            print("TOTAL POSES DETECTED IN PUSH STROKE: " + str(len(eva_push)))
            # checkDistancePose(eva_push)
            biasDataset(my_array_push, eva_push, epsilon_push)
            flatten_arr_bias_push = np.array(epsilon_push)
            # print("Bias: " + str(epsilon_push))
            df = pd.DataFrame(epsilon_push)
            df.to_csv("epsilonPUSH.txt")
            df = pd.DataFrame(eva_push)
            df.to_csv("evaluatePUSH.txt")
            # print("WRONG POSES: " + str(len(findWrongPose(eva_push, wrong_push))))
            # print("TOTAL VALUES: " + str(sumWrongPose(eva_push)))
            # print("TOTAL WRONG VALUES: " + str(sumWrongPose(wrong_push)))
        elif(len(i) == len(lm_list_data_topspin)):
            # print("TOPSPIN: " + str(len(i)))
            print("TOPSPIN")
            # print("TOPSPIN POSITION: " + str(len(lm_list_data_topspin)))
            evaluatePose(my_array_topspin, lm_list_data_topspin, eva_topspin)
            print("TOTAL POSES DETECTED IN TOPSPIN STROKE: " + str(len(eva_topspin)))
            # checkDistancePose(eva_topspin)
            biasDataset(my_array_topspin, eva_topspin, epsilon_topspin)
            flatten_arr_bias_topspin = np.array(epsilon_topspin)
            # print("Bias: " + str(epsilon_topspin))
            df = pd.DataFrame(epsilon_topspin)
            df.to_csv("epsilonTOPSPIN.txt")
            df = pd.DataFrame(eva_topspin)
            df.to_csv("evaluateTOPSPIN.txt")
            # print("WRONG POSES: " + str(len(findWrongPose(eva_topspin, wrong_topspin))))
            # print("TOTAL VALUES: " + str(sumWrongPose(eva_topspin)))
            # print("TOTAL WRONG VALUES: " + str(sumWrongPose(wrong_topspin)))
        elif(len(i) == len(lm_list_data_pushbh)):
            # print("PUSH BACKHAND: " + str(len(i)))
            print("PUSH BACKHAND")
            # print("PUSH BACKHAND POSITION: " + str(len(lm_list_data_pushbh)))
            evaluatePose(my_array_pushbh, lm_list_data_pushbh, eva_pushbh)
            print("TOTAL POSES DETECTED IN PUSH BACKHAND STROKE: " + str(len(eva_pushbh)))
            # checkDistancePose(eva_pushbh)
            biasDataset(my_array_pushbh, eva_pushbh, epsilon_pushbh)
            flatten_arr_bias_pushbh = np.array(epsilon_pushbh)
            # print("Bias: " + str(epsilon_pushbh))
            df = pd.DataFrame(epsilon_pushbh)
            df.to_csv("epsilonPUSHBH.txt")
            df = pd.DataFrame(eva_pushbh)
            df.to_csv("evaluatePUSHBH.txt")
            # print("WRONG POSES: " + str(len(findWrongPose(eva_pushbh, wrong_pushbh))))
            # print("TOTAL VALUES: " + str(sumWrongPose(eva_pushbh)))
            # print("TOTAL WRONG VALUES: " + str(sumWrongPose(wrong_pushbh)))
        elif(len(i) == len(lm_list_data_topspinbh)):
            # print("TOPSPIN BACKHAND: " + str(len(i)))
            print("TOPSPIN BACKHAND")
            # print("TOPSPIN BACKHAND POSITION: " + str(len(lm_list_data_topspinbh)))
            evaluatePose(my_array_topspinbh, lm_list_data_topspinbh, eva_topspinbh)
            # checkDistancePose(eva_topspinbh)
            print("TOTAL POSES DETECTED IN TOPSPIN BACKHAND STROKE: " + str(len(eva_topspinbh)))
            biasDataset(my_array_topspinbh, eva_topspinbh, epsilon_topspinbh)
            # arr_bias_topspinbh = [item for sublist in epsilon_topspinbh for item in sublist]
            flatten_arr_bias_topspinbh = np.array(epsilon_topspinbh)
            # print("Bias: " + str(epsilon_topspinbh))
            df = pd.DataFrame(epsilon_topspinbh)
            df.to_csv("epsilonTOPSPINBH.txt")
            df = pd.DataFrame(eva_topspinbh)
            df.to_csv("evaluateTOPSPINBH.txt")
            # print("WRONG POSES: " + str(len(findWrongPose(eva_topspinbh, wrong_topspinbh))))
            # print("TOTAL VALUES: " + str(sumWrongPose(eva_topspinbh)))
            # print("TOTAL WRONG VALUES: " + str(sumWrongPose(wrong_topspinbh)))

cap.release()
cv2.destroyAllWindows()
