#======================================================================================================================================
#======================================================================================================================================
#======================================================================================================================================
# 2023Sep10 mcvogt 
# USE CASE
# the goal was to develop a script that could be executed at the command line, accept an input [image] file, and pass
# it to a local Azure Custom Vision model [Object Detector] exported as an ONNX model
# this project is for the MN DNR, to provide a public service to all hunters, land owners, and interested wildlife enthusiests to 
# help them recognize when their local deer are infected with CWD.   The model was conceived by mcvogt, and trained with imagery data
# from mcvogt's trail and outdoor cameras.  

# HISTORY/REVISIONS  STACK FASHION PUSHING DOWN 
#======================================================================================
# 2023Nov09 mcvogt
# mike modifying this to try and reach into Azure Storage Account for a file to process...
# test file is in ASA ...   anonymous blob abd container access is permitted...
# THIS is the error we got when trying to just run existing predict.py as-is and only changing local file path/name for this URI...

#FileNotFoundError: [Errno 2] No such file or directory: '/mnt/batch/tasks/shared/LS_root/mounts/clusters/compute-ds11v2-aivision/code/Users/michael.vogt/
#           aivisionexercises/https:/asatestimages01.blob.core.windows.net/container-testimages/DEER_CWD_imagery/Cropped_IC_Model/DeerDayHealthyCandidates/TestImageBlob.jpg'

# would like to Only have this 'https:/asatestimages01.blob.core.windows.net/container-testimages/DEER_CWD_imagery/Cropped_IC_Model/DeerDayHealthyCandidates/TestImageBlob.jpg'

# mike read up on argparser, and commented out all working code in Main() - isolating just the argument parsing...
# mike now has converted this over to expect second argument to be a valid URL to a file, no keys or authorization has been applied to mikes source, to make the URL easier...
# mike fixed the above problem by dropping the type= from the parser call, preventing it from forcing a URL to conform to a filepathname (single /'s only)
#    parser.add_argument('image_filepath', type=pathlib.Path)    # define the 2nd real argument called image_filepath and make sure will be of type pathlib.Path... 
#    parser.add_argument('image_filepath')    # define the 2nd real argument called image_filepath and do NOT validate it using type=

# now, ist working on AML Workspace!!!!!!



#========================================= example ====================================
#======================================================================================

# IMPORTS
import argparse
import pathlib
import numpy as np
import onnx
import onnxruntime
import PIL.Image
#-----------------------------------------------------------------------------------------------------
# from PIL import Image  (is this just a different way to import the same thing?  comment out first time testing)
import requests
#-----------------------------------------------------------------------------------------------------

PROB_THRESHOLD = 0.01  # Minimum probably to show results.

#-------------------------------------------------------------------
class Model:
    def __init__(self, model_filepath):
        self.session = onnxruntime.InferenceSession(str(model_filepath))
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

    def predict(self, image_filepath):  # when this is called by main(), args.image_filepath becomes local image_filepath...
        url = image_filepath  # this is local handle to the image's filepathname...
        image = PIL.Image.open(requests.get(url, stream=True).raw)
        image = image.resize(self.input_shape)  # this works only inside predict.py because class Model is defined here...

        input_array = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
        input_array = input_array.transpose((0, 3, 1, 2))  # => (N, C, H, W)
        if self.is_bgr:
            input_array = input_array[:, (2, 1, 0), :, :]
        if not self.is_range255:
            input_array = input_array / 255  # => Pixel values should be in range [0, 1]

        outputs = self.session.run(self.output_names, {self.input_name: input_array.astype(self.input_type)})
        return {name: outputs[i] for i, name in enumerate(self.output_names)}
#-------------------------------------------------------------------



#------------ helper functions ------------------
# https://www.freecodecamp.org/news/python-switch-statement-switch-case-example/
def switch(class_id):
    if class_id == 0:
        return "Healthy Deer"
    elif class_id == 1:
        return "UnHealthy Deer"

def print_outputs(outputs):
    assert set(outputs.keys()) == set(['detected_boxes', 'detected_classes', 'detected_scores'])
    for box, class_id, score in zip(outputs['detected_boxes'][0], outputs['detected_classes'][0], outputs['detected_scores'][0]):
        if score > PROB_THRESHOLD:
            clabel = switch(class_id)
#           print(f"Label: {class_id}, Probability: {score:.5f}, box: ({box[0]:.5f}, {box[1]:.5f}) ({box[2]:.5f}, {box[3]:.5f})")
            print(f"Label: {clabel}, Probability: {score:.5f}, box: ({box[0]:.5f}, {box[1]:.5f}) ({box[2]:.5f}, {box[3]:.5f})")
#------------ helper functions ------------------


#------------------------------------------------------------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()                          # instantiate the parser...
    parser.add_argument('model_filepath', type=pathlib.Path)    # define the 1st real argument called model_filepath and make sure will be of type pathlib.Path...
    parser.add_argument('image_filepath')                       # define the 2nd real argument called image_filepath and do NOT validate it using type=
                                                                # NOTE - BOTH of these arguments are required, and Must be in this order...
    args = parser.parse_args()

    print() # need a CRLF for clarity during output...
    print("args.model_filepath  = ", args.model_filepath)
    print("args.image_filepath  = ", args.image_filepath)
    print() # need a CRLF

    model = Model(args.model_filepath)            # this creates a model using the model source location argument...

    outputs = model.predict(args.image_filepath)  # this uses the created model and calls predict method passing in the image source URL argument...
    print_outputs(outputs) # 


if __name__ == '__main__':
    main()
#------------------------------------------------------------------------------------------------------------------------------------------------------------------

#======================================================================================================================================
#======================================================================================================================================
#======================================================================================================================================
    