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
        self.eye_surrounding_area = 10
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
            return self.preprocess_output(result[self.output][0])

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
        net_input_shape = self.network.inputs[self.input].shape
        p_frame = cv2.resize(image, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose(2, 0, 1)
        p_frame = p_frame.reshape(1, *p_frame.shape)
        return p_frame


    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        leye_x = outputs[0].tolist()[0][0]
        leye_y = outputs[1].tolist()[0][0]
        reye_x = outputs[2].tolist()[0][0]
        reye_y = outputs[3].tolist()[0][0]
        
        box = (leye_x, leye_y, reye_x, reye_y)

        h, w = self.image.shape[0:2]
        # w = image.shape[1]
        box = box * np.array([w, h, w, h])
        box = box.astype(np.int32)

        (lefteye_x, lefteye_y, righteye_x, righteye_y) = box
        # cv2.rectangle(image,(lefteye_x,lefteye_y),(righteye_x,righteye_y),(255,0,0))

        le_xmin = lefteye_x - self.eye_surrounding_area
        le_ymin = lefteye_y - self.eye_surrounding_area
        le_xmax = lefteye_x + self.eye_surrounding_area
        le_ymax = lefteye_y + self.eye_surrounding_area

        re_xmin = righteye_x - self.eye_surrounding_area
        re_ymin = righteye_y - self.eye_surrounding_area
        re_xmax = righteye_x + self.eye_surrounding_area
        re_ymax = righteye_y + self.eye_surrounding_area

        left_eye = self.image[le_ymin:le_ymax, le_xmin:le_xmax]
        right_eye = self.image[re_ymin:re_ymax, re_xmin:re_xmax]
        eye_coords = [[le_xmin, le_ymin, le_xmax, le_ymax], [re_xmin, re_ymin, re_xmax, re_ymax]]

        return left_eye, right_eye, eye_coords
