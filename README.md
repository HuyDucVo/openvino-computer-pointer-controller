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
The default value to run the program is included with the minimal setting. You can simply run ```python3 main.py``` from the src directory.  
Or you can follow the documentation below to type in a custom command. For example:  
```-f ../models/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 -l ../models/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009 -hp ../models/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001 -g ../models/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002 -i ../bin/demo.mp4 -t video -d CPU -debug 0 -p low -s fast```   
  
For this project, if you want to show:  
- all stats  
```
python3 main.py -flag 0
```
- draw the face  
```
python3 main.py -flag 1
```
- draw the eyes  
```
python3 main.py -flag 2
```
- print the mouse coors  
```
python3 main.py -flag 3
```
## Documentation
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

## Benchmarks
- Model Loading Time report:  
-- FP 32
```
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
```
-- FP16
```
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
```
-- INT8
```
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
```
## Results
Recalling, FP16 and INT8 requires less precision to perform the calculation. Thus, it requires less power for memory and does inference faster. It has a drawback is that not all models we found online is FP16 or INT8, but FP32. Converting to FP16 requires more work and not always work.  
Theorictically, FP16 and INT8 can work 2x compared with FP32.  
Both FP16 and INT8 save memory and could give a significant speedup comparing to FP32.  
Since we are experimenting on a small set for this project, the different may not too obvious. But in the Head Pose Detection Model, we can see the order of Inference Time trend is FP32 -> FP16 -> INT8
Sources:   
https://blog.inten.to/hardware-for-deep-learning-part-3-gpu-8906c1644664#ea08  
https://docs.openvinotoolkit.org/latest/openvino_docs_performance_int8_vs_fp32.html  
### Edge Cases
There are some edge case for people with disability. We can see that it detect the iris of the eyes to determine the vector. And some people have a different kind of eyes ball.  
Another case is we you move the mouse out of the window.  
Last egde case is the OS and environment when do this, I use CPU and VMware to perform this project and it may not grab the image of the mouse. But we can add some code as I put in the file to check if it is actually moving.
