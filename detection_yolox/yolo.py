import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont

from .nets.yolo import YoloBody
from .utils.utils import cvtColor, get_classes, preprocess_input, resize_image
from .utils.utils_bbox import decode_outputs, non_max_suppression

'''
'''
class YOLO(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        
        
        
        #--------------------------------------------------------------------------#
        "model_path"        : 'detection_yolox/model_data/YOLOX-final.pth',
        "classes_path"      : 'detection_yolox/model_data/ship_classes.txt',
        #---------------------------------------------------------------------#
        
        #---------------------------------------------------------------------#
        "input_shape"       : [640, 640],
        #---------------------------------------------------------------------#
        
        #---------------------------------------------------------------------#
        "phi"               : 's',
        #---------------------------------------------------------------------#
        
        #---------------------------------------------------------------------#
        "confidence"        : 0.5,
        #---------------------------------------------------------------------#
        
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.3,
        #---------------------------------------------------------------------#
        
        
        #---------------------------------------------------------------------#
        "letterbox_image"   : True,
        #-------------------------------#
        
        
        #-------------------------------#
        "cuda"              : torch.cuda.is_available(),
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            
        #---------------------------------------------------#
        
        #---------------------------------------------------#
        self.class_names, self.num_classes  = get_classes(self.classes_path)

        #---------------------------------------------------#
        
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

    #---------------------------------------------------#
    
    #---------------------------------------------------#
    def generate(self):
        self.net    = YoloBody(self.num_classes, self.phi)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()

        print('{} model, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    #---------------------------------------------------#
    
    #---------------------------------------------------#
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        
        
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        
        
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = decode_outputs(outputs, self.input_shape)
            #---------------------------------------------------------#
            
            #---------------------------------------------------------#
            results = non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return []

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]
        #---------------------------------------------------------#
        
        #---------------------------------------------------------#

        out = []
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            y1     = max(0, np.floor(top).astype('int32'))
            x1    = max(0, np.floor(left).astype('int32'))
            y2  = min(image.size[1], np.floor(bottom).astype('int32'))
            x2   = min(image.size[0], np.floor(right).astype('int32'))
            
            score_tensor = torch.from_numpy(np.array(score))
            if self.cuda:
                score_tensor = score_tensor.cuda()
            out.append((x1,y1,x2,y2,predicted_class,score_tensor))


        return out

