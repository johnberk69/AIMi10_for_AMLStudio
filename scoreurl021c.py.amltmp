# 11/21/2023 jberk
# this is functioning and tested version of score.py used during Deployment of AML models to Azure Managed Endpoints
# each score.py file must have at least an init() and a run() function.   init() is automatically looked for and
# executed when the host container is spun up, and run() is looked for an executed when any REST request is made to
# the Endpoint hosting the model.

# this code, Deployment, and Endpoint host Iteration 10 of the Animal Identification Model for Bushnell

import json  
import os
import PIL.Image  
import requests  

from io import BytesIO
from azureml.contrib.services.aml_request import AMLRequest, rawhttp   #...
from azureml.contrib.services.aml_response import AMLResponse

import onnx
import onnxruntime

import numpy as np
from numpy import array     # 
from numpy import float32   # 
from numpy import int64     # 

import time
import sys
from azureml.core.model import Model

PROB_THRESHOLD = 0.01  


class Model:
    def __init__(self, model_filepath):
        self.session = onnxruntime.InferenceSession(
    #       modelPath, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']  # will leave in place for when CUDA/GPU might be available  
            modelPath, providers=['CPUExecutionProvider']  
        )
        assert len(self.session.get_inputs()) == 1
        self.input_shape = self.session.get_inputs()[0].shape[2:]
        self.input_name = self.session.get_inputs()[0].name
        self.input_type = {'tensor(float)': np.float32, 'tensor(float16)': np.float16}[self.session.get_inputs()[0].type]
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.is_bgr = False
        self.is_range255 = False
        onnx_model = onnx.load(model_filepath)
        for metadata in onnx_model.metadata_props:
            if metadata.key == 'Image.BitmapPixelFormat' and metadata.value == 'Bgr8':
                self.is_bgr = True
            elif metadata.key == 'Image.NominalPixelRange' and metadata.value == 'NominalRange_0_255':
                self.is_range255 = True

    def predict(self, image_filepath):  
        url = image_filepath  
        image = PIL.Image.open(requests.get(url, stream=True).raw) 
        image = image.resize(self.input_shape)  

        input_array = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
        input_array = input_array.transpose((0, 3, 1, 2))  # => (N, C, H, W)
        if self.is_bgr:
            input_array = input_array[:, (2, 1, 0), :, :]
        if not self.is_range255:
            input_array = input_array / 255  

        outputs = self.session.run(self.output_names, {self.input_name: input_array.astype(self.input_type)})
        return {name: outputs[i] for i, name in enumerate(self.output_names)}


def switch(class_id):  # this switch function needs to be updated for each implementation of this code to match a given model.  future versions should read and process labels.txt for this
    if class_id == 0:
        return "BearDay"
    elif class_id == 1:
        return "BearNight"
    elif class_id == 2:
        return "CougarDay"
    elif class_id == 3:
        return "CougarNight"
    elif class_id == 4:
        return "CoyoteDay"
    elif class_id == 5:
        return "CoyoteNight"
    elif class_id == 6:
        return "CraneDay"
    elif class_id == 7:
        return "DeerBuckDay"
    elif class_id == 8:
        return "DeerBuckNight"
    elif class_id == 9:
        return "DeerDoeDay"
    elif class_id == 10:
        return "DeerDoeNight"
    elif class_id == 11:
        return "ElkBullDay"
    elif class_id == 12:
        return "ElkBullNight"
    elif class_id == 13:
        return "ElkCowDay"
    elif class_id == 14:
        return "ElkCowNight"
    elif class_id == 15:
        return "HumanDay"
    elif class_id == 16:
        return "HumanNight"
    elif class_id == 17:
        return "MooseBullDay"
    elif class_id == 18:
        return "MooseBullNight"
    elif class_id == 19:
        return "MooseCowDay"
    elif class_id == 20:
        return "MooseCowNight"
    elif class_id == 21:
        return "SwineDay"
    elif class_id == 22:
        return "SwineNight"
    elif class_id == 23:
        return "TurkeyDay"
    elif class_id == 24:
        return "TurkeyNight"
    elif class_id == 25:
        return "VehicleDay"
    elif class_id == 26:
        return "VehicleNight"
    elif class_id == 27:
        return "WolfDay"
    elif class_id == 28:
        return "WolfNight"

def print_outputs(outputs):
    assert set(outputs.keys()) == set(['detected_boxes', 'detected_classes', 'detected_scores'])
    for box, class_id, score in zip(outputs['detected_boxes'][0], outputs['detected_classes'][0], outputs['detected_scores'][0]):
        if score > PROB_THRESHOLD:
            clabel = switch(class_id)
            print(f"Label: {clabel}, Probability: {score:.5f}, box: ({box[0]:.5f}, {box[1]:.5f}) ({box[2]:.5f}, {box[3]:.5f})")


def init():
    print("entering init()")
    global modelPath   
    global model     

    modelPath = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'TCAI_AIM_Iteration_10_on_Gen_Compact_S1.ONNX/model.onnx') # this must be edited to match the Registered model in AML Studio             
    print("modelPath: ", modelPath) 

    model = Model(modelPath)
    print("created object model of Model class inside score's init()...")

    print("creating model object also created a model.session...")
    print("exiting init()")



@rawhttp   
def run(request):         
    print("entering run()...")
    if request.method == 'POST':
        print("entering POST request handler loop...")
        dictData = request.get_json()  
        print("extracting dictData from POST request = ", dictData) 
        image_url = dictData["image_url"] 
        print("extracting image_url value using key = ", image_url) 

        try:
            outputs = model.predict(image_url)  
            print("called model.predict() inside of run()...")

            print("outputs is of type: ", type(outputs))
            print("[raw] outputs = ", outputs)
            print_outputs(outputs) 

            outputsFL = {} 
            outputsFL["detected_boxes"] = outputs["detected_boxes"].flatten().tolist()
            outputsFL["detected_classes"] = outputs["detected_classes"].flatten().tolist()
            outputsFL["detected_scores"] = outputs["detected_scores"].flatten().tolist()
            json_outputsFL = json.dumps(outputsFL)
            print("json serialization of outputsFL = ", json_outputsFL)

        except Exception as ex:
            outputs = "exception thrown for model.predict(). see error below."
            print(ex)

        print("done with if POST loop...")
        print("Exiting run()...")

        return json_outputsFL    
    else:
        return AMLResponse("bad request, use POST", 500)
