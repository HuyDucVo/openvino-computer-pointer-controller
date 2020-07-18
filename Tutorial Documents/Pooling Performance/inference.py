from openvino.inference_engine import IENetwork, IECore

import numpy as np
import time

# Getting model bin and xml file
model_path='pool_cnn/pool_cnn'
model_weights=model_path+'.bin'
model_structure=model_path+'.xml'

# TODO: Load the model
# Use either IECore or IEPlugin API
model=IENetwork(model_structure, model_weights)


# Reading and Preprocessing Image
input_img=np.load('image.npy')
input_img=input_img.reshape(1, 28, 28)
core = IECore()
net = core.load_network(network=model, device_name='CPU', num_requests=1)


# TODO: Using the input image, run inference on the model for 10 iterations
input_name=next(iter(model.inputs))
input_dict={input_name:input_img}

start=time.time()
for _ in range(10):
    net.infer(input_dict)


print("Time taken to run 10 iterations is: {} seconds".format(time.time()-start))

# TODO: Finish the print statement
print("Time taken to run 10 iterations is: ")