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

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-f", "--face", required=False, type=str,
                        help="Face detection model directory",
                        default='../models/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001')
    parser.add_argument("-l", "--landmarks", required=False, type=str,
                        help="Facial Landmark Detection model directory",
                        #default='../models/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009')
                        #default='../models/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009')
                        default='../models/landmarks-regression-retail-0009/FP16-INT8/landmarks-regression-retail-0009')
    parser.add_argument("-hp", "--headpose", required=False, type=str,
                        help="Head Pose Estimation model directory",
                        #default='../models/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001')
                        #default='../models/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001')
                        default='../models/head-pose-estimation-adas-0001/FP16-INT8/head-pose-estimation-adas-0001')
    parser.add_argument("-g", "--gazeestimation", required=False, type=str,
                        help="Gaze Estimation model directory",
                        #default='../models/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002')
                        #default='../models/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002')
                        default='../models/gaze-estimation-adas-0002/FP16-INT8/gaze-estimation-adas-0002')
    parser.add_argument("-i", "--input", required=False, type=str,
                        help="Input file directory",
                        default='../bin/demo.mp4')
    parser.add_argument("-t", "--input_type", required=False, default='video', type=str,
                        help="Source type is " + 'video' + "|" + 'webcam' + " | " + 'image')
    parser.add_argument("-flag", "--flag", required=False, type=str,
                        help="Partial debug each part of inference pipeline. 0 : all stats report | 1 : draw face detected | 2 : draw eyes detected | 3 : print coors of mouse with gaze estimation and move the mouse",
                        default='4')
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
    activate_frame_count = 10
    if args.input_type == 'video' or args.input_type == 'image':
        feeder = InputFeeder(args.input_type, args.input)
    elif args.input_type == 'webcam':
        feeder = InputFeeder(args.input_type, args.input)
    else:
        print("Input not found. Exit")
        exit(1)

    
    mouse_controller = MouseController(args.precision, args.speed)


    feeder.load_data()
    start_time = 0

    face_model_load_time = 0
    start_time = time.time()
    face_model = FaceDetection(args.face, args.device, args.cpu_extension)
    face_model.load_model()
    face_model_load_time = time.time()-start_time
    print("Face Detection Model Loaded...")

    head_pose_estimation_load_time = 0
    start_time = time.time()
    head_pose_estimation = HeadPoseEstimation(args.headpose, args.device, args.cpu_extension)
    head_pose_estimation.load_model()
    head_pose_estimation_load_time = time.time()-start_time
    print("Head Pose Detection Model Loaded...")

    facial_landmarks_detection_load_time = 0
    start_time = time.time()
    facial_landmarks_detection = FacialLandmarksDetection(args.landmarks, args.device, args.cpu_extension)
    facial_landmarks_detection.load_model()
    facial_landmarks_detection_load_time = time.time()-start_time
    print("Facial Landmark Detection Model Loaded...")

    gaze_model_load_time = 0
    start_time = time.time()
    gaze_model = GazeEstimation(args.gazeestimation, args.device, args.cpu_extension)
    gaze_model.load_model()
    gaze_model_load_time = time.time()-start_time
    print("Gaze Estimation Model Loaded...")

    frame_count = 0

    total_face_model_inference_time = 0
    total_head_pose_estimation_inference_time = 0
    total_facial_landmarks_detection_inference_time = 0
    total_gaze_model_inference_time = 0
    start_time = 0
    for frame in feeder.next_batch():
        if frame is None:
            break
        frame_count += 1
        key = cv2.waitKey(60)

        start_time = time.time()
        first_face_box, first_face = face_model.predict(frame.copy())
        total_face_model_inference_time = total_face_model_inference_time + (time.time() - start_time)
        
        start_time = time.time()
        head_pose_output = head_pose_estimation.predict(first_face_box.copy())
        total_head_pose_estimation_inference_time = total_head_pose_estimation_inference_time + (time.time() - start_time)

        start_time = time.time()
        left_eye, right_eye, eye_coords= facial_landmarks_detection.predict(first_face_box.copy())
        total_facial_landmarks_detection_inference_time = total_facial_landmarks_detection_inference_time + (time.time() - start_time)

        start_time = time.time()
        move_to_coors_mouse = gaze_model.predict(left_eye, right_eye, head_pose_output)
        total_gaze_model_inference_time = total_gaze_model_inference_time + (time.time() - start_time)

        if frame_count%activate_frame_count==0 and (args.flag == "3" or args.flag == "4"):
            mouse_controller.move(move_to_coors_mouse[0],move_to_coors_mouse[1])
            cv2.imshow('video',frame)   
            key = cv2.waitKey(60) 
        if key==27:
            break

        if args.flag == "1":
            cv2.rectangle(frame,(first_face[0],first_face[1] ), (first_face[2],first_face[3]),(255,0,0))
            cv2.imshow('video',frame)
            key = cv2.waitKey(60)
        elif args.flag == "2":
            cv2.rectangle(facial_landmarks_detection.image,(eye_coords[0],eye_coords[1]),(eye_coords[2],eye_coords[3]),(255,0,0))
            cv2.imshow('video',facial_landmarks_detection.image)
            key = cv2.waitKey(60)
        elif args.flag == "3":
            if frame_count == 1:
                print("Printing mouse coors: ") 
            print(move_to_coors_mouse)


    #Report
    if args.flag == "0":
        print('------------- BEGIN REPORT -------------')
        avg_inference_face_model = total_face_model_inference_time / frame_count
        avg_inference_headpose = total_head_pose_estimation_inference_time / frame_count
        avg_inference_facial_landmark = total_facial_landmarks_detection_inference_time / frame_count
        avg_inference_gaze_model = total_gaze_model_inference_time / frame_count

        print("Face Detection Model Load Time: ", args.face)
        print("Loading time: " , face_model_load_time )
        print("Inference time: " , avg_inference_face_model )

        print("Head Pose Detection Model: ", args.headpose)
        print("Loading time: ", head_pose_estimation_load_time )
        print("Inference time:", avg_inference_headpose)

        print("Facial Landmark Detection Model Load Time: ", args.landmarks)
        print("Loading time: ", facial_landmarks_detection_load_time )
        print("Inference time:", avg_inference_facial_landmark)

        print("Gaze Estimation Model Load Time: ", args.gazeestimation)
        print("Loading time: ", gaze_model_load_time )
        print("Inference time:", avg_inference_gaze_model)
        
        
        print('------------- END REPORT -------------')
        

if __name__ == '__main__':
    #arg = '-f ../models/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 -l ../models/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009 -hp ../models/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001 -ge ../models/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002 -i ../bin/demo.mp4 -it video -d CPU -debug headpose gaze face'.split(' ')
    args = build_argparser().parse_args()
    try:
        test_run(args) 
    except:
        e = sys.exc_info()[0]
        print(e)
    finally:
        print("Computer Pointer Controller Finished")