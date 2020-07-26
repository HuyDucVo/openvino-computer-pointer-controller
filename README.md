# Computer Pointer Controller

A Computer Vision app built with OpenVino library that can control the PC mouse by the direction of eyes from image, camera or video. 

## Project Set Up and Installation
Install all the requirements from the requirements.txt
Directories:  
- /src: The models caller, mouse controller, and main file for test run  
- /models: The actual implemented model .xml and .bin files  
- /bin: The input media for video  
- /Tutorial Documents: All the tutorial for further use  <br><br>
- requirements.txt: All the core dependencies are here    <br><br>
Models:
Please use model downloader from openvino on you end to download these model below and put them in /models directory. You can re-download the model or update the model.  
- Face Detection: https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html  
- Head Pose Estimation: https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html  
- Facial Landmarks Detection: https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html  
- Gaze Estimation Model: https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html  

## Demo
The default value to run the program is included with the minimal setting. You can simply run 'python3 main.py' from the src directory.  
Or you can follow the documentation below to type in a custom command. For example:  
'-f ../models/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 -l ../models/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009 -hp ../models/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001 -g ../models/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002 -i ../bin/demo.mp4 -t video -d CPU -debug 0 -p low -s fast'  

## Documentation
'''
-f : --face: Face detection model directory  
-l : --landmarks: Facial Landmark Detection model directory  
-hp : --headpose: Head Pose Estimation model directory  
-g : --gazeestimation: Gaze Estimation model directory  
-i : --input: Input file directory  
-t : --input_type: Source type is video or webcam or image  
-flag : --flag: Partial debug each part of inference pipeline. 0 : all stats report | 1 : draw face detected | 2 : draw eyes detected | 3 : print coors of mouse with gaze estimation  
-ld : --cpu_extension: CPU Extension  
-d : --device: Target device: CPU, GPU, VPU, FPGA  
-p : --precision: Precision Mouse Controller  
-s : --speed: Speed Mouse Controller  
  
--help: Helping menu
'''

## Benchmarks
- Model Loading Time report:  
-- FP16
Face Detection Model Load Time:  ../models/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001
Loading time:  0.09444904327392578
Inference time:  0.01121175491203696
Head Pose Detection Model:  ../models/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001
Loading time:  0.06176471710205078
Inference time: 0.0012917437795865333
Facial Landmark Detection Model Load Time:  ../models/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009
Loading time:  0.051276445388793945
Inference time: 0.0006316395129187633
Gaze Estimation Model Load Time:  ../models/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002
Loading time:  0.08012700080871582
Inference time: 0.0012575771849034196
-- FP 32
Face Detection Model Load Time:  ../models/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001
Loading time:  0.09214973449707031
Inference time:  0.011434975316969015
Head Pose Detection Model:  ../models/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001
Loading time:  0.05920982360839844
Inference time: 0.00131635746713412
Facial Landmark Detection Model Load Time:  ../models/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009
Loading time:  0.051842451095581055
Inference time: 0.0006059266753115896
Gaze Estimation Model Load Time:  ../models/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002
Loading time:  0.07913541793823242
Inference time: 0.0012479838678392313
-- INT8
Face Detection Model Load Time:  ../models/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001
Loading time:  0.09600019454956055
Inference time:  0.011204444755942134
Head Pose Detection Model:  ../models/head-pose-estimation-adas-0001/FP16-INT8/head-pose-estimation-adas-0001
Loading time:  0.10241842269897461
Inference time: 0.0010039725546109473
Facial Landmark Detection Model Load Time:  ../models/landmarks-regression-retail-0009/FP16-INT8/landmarks-regression-retail-0009
Loading time:  0.06687498092651367
Inference time: 0.0005788762690657276
Gaze Estimation Model Load Time:  ../models/gaze-estimation-adas-0002/FP16-INT8/gaze-estimation-adas-0002
Loading time:  0.134962797164917
Inference time: 0.0009797629663499735
## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
