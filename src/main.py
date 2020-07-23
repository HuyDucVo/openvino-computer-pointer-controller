import cv2
import os
import numpy as np

from argparse import ArgumentParser
from gaze_estimation import GazeEstimation
from face_detection import FaceDetection
from mouse_controller import MouseController
from input_feeder import InputFeeder


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-f", "--face", required=True, type=str,
                        help="Path to .xml file of Face Detection model.")
    parser.add_argument("-l", "--landmarks", required=True, type=str,
                        help="Path to .xml file of Facial Landmark Detection model.")
    parser.add_argument("-hp", "--headpose", required=True, type=str,
                        help="Path to .xml file of Head Pose Estimation model.")
    parser.add_argument("-ge", "--gazeestimation", required=True, type=str,
                        help="Path to .xml file of Gaze Estimation model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to video file or enter cam for webcam")
    parser.add_argument("-it", "--input_type", required=True, type=str,
                        help="Provide the source of video frames." + 'video' + " " + 'webcam' + " | " + 'image')
    parser.add_argument("-debug", "--debug", required=False, type=str, nargs='+',
                        default=[],
                        help="To debug each model's output visually, type the model name with comma seperated after --debug")
    parser.add_argument("-ld", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="linker libraries if have any")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Provide the target device: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable.")

    return parser

def test_run(args):
    feeder = None
    if args.input_type == 'video' or args.input_type == 'image':
        feeder = InputFeeder(args.input_type, args.input)


    mc = MouseController("low", "fast")


    feeder.load_data()

    face_model = FaceDetection(args.face, args.device, args.cpu_extension)
    face_model.load_model()
    print("Face Detection Model Loaded...")

    gaze_model = GazeEstimation(args.gazeestimation, args.device, args.cpu_extension)
    gaze_model.load_model()
    print("Gaze Estimation Model Loaded...")

    frame_count = 0

    for frame in feeder.next_batch():
        if frame is None:
            break
        frame_count += 1

        croppedFace, face_coords = face_model.predict(frame.copy())
        if frame_count%5==0:
            
            #print(croppedFace)
            cv2.imshow('video',cv2.resize(frame,(500,500)))
        
        key = cv2.waitKey(1)
        
    
if __name__ == '__main__':
    #arg = '-f ../models/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 -l ../models/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009 -hp ../models/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001 -ge ../models/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002 -i ../bin/demo.mp4 -it video -d CPU -debug headpose gaze face'.split(' ')
    args = build_argparser().parse_args()
    test_run(args) 