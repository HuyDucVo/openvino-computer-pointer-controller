'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import numpy as np
from openvino.inference_engine import IECore, IENetwork
import math
class FacialLandmarksDetection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        #From params
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.extensions = extensions

        #For application
        self.core = None
        self.network = None
        self.exec_net = None
        self.unsupported_layers = None
        self.image = None

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.core = IECore()
        self.network = self.core.read_network(model=self.model_structure, weights=self.model_weights)
        supported_layers = self.core.query_network(network=self.network, device_name=self.device)
        self.unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]

        self.check_model()
        print("Checked facial-landmark-dection model")

        self.exec_net = self.core.load_network(network=self.network, device_name=self.device,num_requests=1)
        self.input = next(iter(self.network.inputs))
        self.output = next(iter(self.network.outputs))

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        img_processed = self.preprocess_input(image.copy())
        self.exec_net.start_async(request_id= 0, inputs={self.input: img_processed})
        while self.exec_net.requests[0].wait(-1) == 0:
            result = self.exec_net.requests[0].outputs[self.output]
            self.image = image
            return self.preprocess_output(result[0])

    def check_model(self):
        if len(self.unsupported_layers)!=0 :
            self.core.add_extension(self.extensions, self.device)
            supported_layers = self.core.query_network(network = self.network, device_name=self.device)
            self.unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
            if len(self.unsupported_layers)!=0:
                print("Unsupported layers")
                exit(1)

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        net_input_shape = []
        net_input_shape = self.network.inputs[self.input].shape
        p_frame = None
        p_frame = cv2.resize(image, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose(2, 0, 1)
        p_frame = p_frame.reshape(1, *p_frame.shape)
        return p_frame


    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        both_eye_coors = []
        both_eye_coors.append(outputs[0].tolist()[0][0]*self.image.shape[1])
        both_eye_coors.append(outputs[1].tolist()[0][0]*self.image.shape[0])
        both_eye_coors.append(outputs[2].tolist()[0][0]*self.image.shape[1])
        both_eye_coors.append(outputs[3].tolist()[0][0]*self.image.shape[0])
        both_eye_coors = [round(x) for x in both_eye_coors]
        return self.image[both_eye_coors[1]-20 : both_eye_coors[1]+20 , both_eye_coors[0]-20:both_eye_coors[0]+20],self.image[both_eye_coors[3]-20 : both_eye_coors[3]+20 , both_eye_coors[2]-20:both_eye_coors[2]+20], both_eye_coors