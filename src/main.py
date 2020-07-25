import cv2
import os
import numpy as np
import time

from argparse import ArgumentParser
from gaze_estimation import GazeEstimation
from face_detection import FaceDetection
from head_pose_estimation import HeadPoseEstimation
from facial_landmarks_detection import FacialLandmarksDetection
from mouse_controller import MouseController
from input_feeder import InputFeeder

def build_report(flag):
    if flag == 0:
        print()
def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-f", "--face", required=False, type=str,
                        help="Face detection model directory",
                        default='../models/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001')
    parser.add_argument("-l", "--landmarks", required=False, type=str,
                        help="Facial Landmark Detection model directory",
                        default='../models/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009')
    parser.add_argument("-hp", "--headpose", required=False, type=str,
                        help="Head Pose Estimation model directory",
                        default='../models/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001')
    parser.add_argument("-g", "--gazeestimation", required=False, type=str,
                        help="Gaze Estimation model directory",
                        default='../models/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002')
    parser.add_argument("-i", "--input", required=False, type=str,
                        help="Input file directory",
                        default='../bin/demo.mp4')
    parser.add_argument("-t", "--input_type", required=False, default='video', type=str,
                        help="Source type is " + 'video' + "|" + 'webcam' + " | " + 'image')
    parser.add_argument("-debug", "--debug", required=False, type=str, nargs='+',
                        default='0',
                        help="Partial debug each part of inference pipeline. 0 : all stats report | 1 : draw face detected | 2 : draw eyes detected | 3 : print coors of mouse with gaze estimation)
    parser.add_argument("-ld", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="CPU Extension")
    parser.add_argument("-d", "--device", type=str, required=False, default="CPU",
                        help="Target device: CPU, GPU, VPU, FPGA")
    parser.add_argument("-p", "--precision", type=str, required=False, default="low",
                        help="Precision Mouse Controller")
    parser.add_argument("-s", "--speed", type=str, required=False, default="fast",
                        help="Speed Mouse Controller")
    return parser

def test_run(args):
    feeder = None
    if args.input_type == 'video' or args.input_type == 'image':
        feeder = InputFeeder(args.input_type, args.input)
    elif args.input_type == 'webcam':
        feeder = InputFeeder(args.input_type, args.input)
    else:
        print("Input not found. Exit")
        exit(1)

    
    mouse_controller = MouseController(args.precision, args.speed)


    feeder.load_data()

    face_model_load_time = 0
    face_model = FaceDetection(args.face, args.device, args.cpu_extension)
    face_model.load_model()
    print("Face Detection Model Loaded...")

    head_pose_estimation = HeadPoseEstimation(args.headpose, args.device, args.cpu_extension)
    head_pose_estimation.load_model()
    print("Head Pose Detection Model Loaded...")

    facial_landmarks_detection = FacialLandmarksDetection(args.landmarks, args.device, args.cpu_extension)
    facial_landmarks_detection.load_model()
    print("Facial Landmark Detection Model Loaded...")

    gaze_model = GazeEstimation(args.gazeestimation, args.device, args.cpu_extension)
    gaze_model.load_model()
    print("Gaze Estimation Model Loaded...")


    frame_count = 0

    for frame in feeder.next_batch():
        if frame is None:
            break
        frame_count += 1
        key = cv2.waitKey(60)
        croppedFace, face_coords = face_model.predict(frame.copy())
        
        
        head_pose_output = head_pose_estimation.predict(croppedFace.copy())
        
        left_eye, right_eye, eye_coords = facial_landmarks_detection.predict(croppedFace.copy())
        #cv2.imshow('video',facial_landmarks_detection.image)
        #key = cv2.waitKey(60)
    
        new_mouse_coord, gaze_vector = gaze_model.predict(left_eye, right_eye, head_pose_output)
        if frame_count%10==0:
            #Testing the face detection here
            #frame = cv2.rectangle(frame, (face_coords[0],face_coords[1] ), (face_coords[2],face_coords[3]),  (255,0,0) , -1 )
            #cv2.imshow('video',frame)
            #mouse_controller.move(new_mouse_coord[0],new_mouse_coord[1])    
            print(new_mouse_coord)
        if key==27:
            break
    
    

if __name__ == '__main__':
    #arg = '-f ../models/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 -l ../models/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009 -hp ../models/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001 -ge ../models/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002 -i ../bin/demo.mp4 -it video -d CPU -debug headpose gaze face'.split(' ')
    args = build_argparser().parse_args()
    test_run(args) 