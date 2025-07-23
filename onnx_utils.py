import cv2
import numpy as np
import onnxruntime
from yolov7s.common import letterbox, preprocess, onnx_inference, post_process

cuda = False

class YOLODetect:
    def __init__(self, opt):
        self.conf_thres = opt.conf_thres
        self.init_onnx_model(opt)
        
    def init_onnx_model(self, opt):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(opt.yolo_onnx_path, providers=providers)
        IN_IMAGE_H = self.session.get_inputs()[0].shape[2]
        IN_IMAGE_W = self.session.get_inputs()[0].shape[3]
        self.new_shape = (IN_IMAGE_W, IN_IMAGE_H)
       
    def inference_(self, frame):
        ori_images = [frame.copy()]
        resized_image, ratio, dwdh = letterbox(frame, new_shape=self.new_shape, auto=False)
        input_tensor = preprocess(resized_image)
        outputs = onnx_inference(self.session, input_tensor)
        #pred_output = post_process(outputs, ori_images, ratio, dwdh, self.conf_thres)
        return outputs, ori_images, ratio, dwdh, self.conf_thres

    
    

    
