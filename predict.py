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

# HISTORY/REVISIONS
#======================================================================================
# 2023Nov03 mcvogt
# this is mikes most recent update from inside an Azure ML Notebook - just prior to downloading the whole
# python project with renamed .ONNX folder, so it can be used as source for Deploying to a Managed Endpoint using the 
# AML Studio UI for such things.  that UI looks for a model in the local file system, but apparently not from within AML itself.
# no changes have been made to the model - it will use an inference script (renamed from predictONNX1OD.py back --> predict.py) to be 
# more consistent with original example.  also, it was learned that when deploying to a Triton host, that host does Not require a
# scoring script or environment... so, maybe this has been the confusion for a while.. because other hosts asked for such things.

# 2023Oct31 mcovgt
# note- when running this predict.py script under Azure ML - the DS11v2 conpute is an Anaconda platform, so the venvs are actually cenvs, and 
# can be listed and activated/deactivated using conda activate xxxxx, and conda deactivate
# $ conda env list<enter>
# $ conda deactivate
# $ conda activate nameofdesiredcondaenvironment      # actually switches cenvs without having to deactivate one first...

# (azureml_py38) azureuser@compute-ds11v2-aivision:~/cloudfiles/code/Users/michael.vogt$ conda env list
# # conda environments:
# #
# base                     /anaconda
# azureml_py310_sdkv2      /anaconda/envs/azureml_py310_sdkv2
# azureml_py38          *  /anaconda/envs/azureml_py38
# azureml_py38_PT_TF       /anaconda/envs/azureml_py38_PT_TF
# jupyter_env              /anaconda/envs/jupyter_env

# (azureml_py38) azureuser@compute-ds11v2-aivision:~/cloudfiles/code/Users/michael.vogt$ conda activate azureml_py310_sdkv2
# (azureml_py310_sdkv2) azureuser@compute-ds11v2-aivision:~/cloudfiles/code/Users/michael.vogt$ 

# NOTE - interestingly, THIS session, the compute appears to have retained the earlier-installed ONNX and ONNXRUNTIME packages.  the same ones
# that appeared to have disappeared from the initial development work.  perhaps mike only made an error when moving this project across
# multiple development/execute environments...

# ...
# msrestazure               0.6.4                    pypi_0    pypi
# ncurses                   6.4                  h6a678d5_0  
# ndg-httpsclient           0.5.1                    pypi_0    pypi
# nest-asyncio              1.5.6              pyhd8ed1ab_0    conda-forge
# networkx                  3.1                      pypi_0    pypi
# numpy                     1.25.0                   pypi_0    pypi
# numpy-base                1.26.0          py310hb5e798b_0  
# oauthlib                  3.2.2                    pypi_0    pypi
# onnx                      1.13.1          py310h12ddb61_0  
# onnxruntime               1.15.1          py310hf70ce4d_0  
# opencensus                0.11.2                   pypi_0    pypi
# opencensus-context        0.1.3                    pypi_0    pypi
# opencensus-ext-azure      1.1.9                    pypi_0    pypi
# openssl                   3.0.11               h7f8727e_2  
# packaging                 23.0                     pypi_0    pypi
# pandas                    2.0.2                    pypi_0    pypi
# paramiko                  3.2.0                    pypi_0    pypi
# parso                     0.8.3              pyhd8ed1ab_0    conda-forge
# pathspec                  0.11.1                   pypi_0    pypi
# pexpect                   4.8.0              pyh1a96a4e_2    conda-forge
# pickleshare               0.7.5                   py_1003    conda-forge
# pillow                    9.5.0                    pypi_0    pypi
# pip                       23.1.2          py310h06a4308_0  
# pkginfo                   1.9.6                    pypi_0    pypi
# ...

# # some re/validation
# (azureml_py310_sdkv2) azureuser@compute-ds11v2-aivision:~/cloudfiles/code/Users/michael.vogt/aivisionexercises$ python --version
# Python 3.10.11

# (azureml_py310_sdkv2) azureuser@compute-ds11v2-aivision:~/cloudfiles/code/Users/michael.vogt/aivisionexercises$ python predict.py NADeerCWDclassifier.ONNX/model.ONNX TestImageHealthyDeerDayBuck.jpg 
# Label: Healthy Deer, Probability: 0.99009, box: (0.29941, 0.50782) (0.61335, 0.81000)
# Label: UnHealthy Deer, Probability: 0.02532, box: (0.34874, 0.00170) (0.58188, 0.55292)
# Label: Healthy Deer, Probability: 0.01254, box: (0.00925, 0.15533) (0.87207, 0.98420)
# (azureml_py310_sdkv2) azureuser@compute-ds11v2-aivision:~/cloudfiles/code/Users/michael.vogt/aivisionexercises$ 

# (azureml_py310_sdkv2) azureuser@compute-ds11v2-aivision:~/cloudfiles/code/Users/michael.vogt/aivisionexercises$ python predict.py NADeerCWDclassifier.ONNX/model.ONNX TestImageUnHealthyDeerDayBuck.jpg 
# Label: UnHealthy Deer, Probability: 0.13009, box: (0.07849, 0.59303) (0.71738, 0.99858)
# Label: Healthy Deer, Probability: 0.03130, box: (0.04158, 0.04641) (0.94205, 0.97782)
# Label: UnHealthy Deer, Probability: 0.01827, box: (0.55772, 0.04974) (0.78629, 0.40335)
# Label: UnHealthy Deer, Probability: 0.01671, box: (0.22403, 0.21533) (0.77413, 0.71315)

# (azureml_py310_sdkv2) azureuser@compute-ds11v2-aivision:~/cloudfiles/code/Users/michael.vogt/aivisionexercises$ 







# 2023Oct27 mcvogt
# this version of the CWD model was moved into one of mike's Azure ML workspaces and executed correctly from the terminal there using only a DS11v2 2-core CPU
# the default pythin 3.8 installed on that compute was already pre-configured with all needed packages to run imported ONNX models... NOTHING was installed using pip
# example output below...    note - file structure in compute-local ubuntu 20.04 file system matched mikes Dell development and VirtualBox7.0 Ubuntu file systems...

# drwxrwxrwx 2 root root      0 Oct 27 20:25 .
# drwxrwxrwx 2 root root      0 Oct 27 20:17 ..
# -rwxrwxrwx 1 root root    315 Oct 27 20:26 .amlignore
# -rwxrwxrwx 1 root root    315 Oct 27 20:26 .amlignore.amltmp
# drwxrwxrwx 2 root root      0 Oct 27 20:29 41c20ea0f5414a3381bfab173c1b97a2.ONNX
# -rwxrwxrwx 1 root root 366147 Oct 27 20:26 TestImageHealthyDeerDayBuck.jpg
# -rwxrwxrwx 1 root root 371809 Oct 27 20:26 TestImageUNHealthyDeerDayBuck.jpg
# -rwxrwxrwx 1 root root   7802 Oct 27 20:26 predictONNXS1OD.py
# -rwxrwxrwx 1 root root   2675 Oct 27 20:26 requirements.txt
# (azureml_py38) azureuser@compute-ds11v2-aivision:~/cloudfiles/code/Users/michael.vogt/aivisionexercises$ python --version
# Python 3.8.5

# (azureml_py38) azureuser@compute-ds11v2-aivision:~/cloudfiles/code/Users/michael.vogt/aivisionexercises$ python predictONNXS1OD.py 41c20ea0f5414a3381bfab173c1b97a2.ONNX//model.onnx TestImageHealthyDeerDayBuck.jpg 
# Label: Healthy Deer, Probability: 0.99026, box: (0.29925, 0.50730) (0.61337, 0.81005)
# Label: UnHealthy Deer, Probability: 0.02626, box: (0.34982, 0.00156) (0.58259, 0.55344)
# Label: Healthy Deer, Probability: 0.01254, box: (0.01045, 0.15314) (0.87271, 0.98387)
# (azureml_py38) azureuser@compute-ds11v2-aivision:~/cloudfiles/code/Users/michael.vogt/aivisionexercises$ 

# (azureml_py38) azureuser@compute-ds11v2-aivision:~/cloudfiles/code/Users/michael.vogt/aivisionexercises$ python predictONNXS1OD.py 41c20ea0f5414a3381bfab173c1b97a2.ONNX//model.onnx TestImageUnHealthyDeerDayBuck.jpg 
# Label: UnHealthy Deer, Probability: 0.13061, box: (0.07830, 0.59306) (0.71790, 0.99864)
# Label: Healthy Deer, Probability: 0.03156, box: (0.04017, 0.04701) (0.94438, 0.97847)
# Label: UnHealthy Deer, Probability: 0.01732, box: (0.55717, 0.05044) (0.78642, 0.40183)
# Label: UnHealthy Deer, Probability: 0.01644, box: (0.22265, 0.21637) (0.77419, 0.71406)
# (azureml_py38) azureuser@compute-ds11v2-aivision:~/cloudfiles/code/Users/michael.vogt/aivisionexercises$ 

# 2023Oct17 mcvogt
# follow up and documenting.  migrating now to a VM hosted by VirtualBox under Wind11Pro
# Win11Pro(host)\VirtualBox7.0\Ubuntu20.04LTS(guest)\Python3.11.03\this Script

# 2023Oct09 mcvogt
# later improved code to format Category/Class Label in human-friendly form... 
# (venv-python3113-datascience) C:\Development\GitHub\AIVisionExercises>python predictONNXS1OD.py 41c20ea0f5414a3381bfab173c1b97a2.ONNX/model.onnx TestImageHealthyDeerDayBuck.jpg
# Label: Healthy Deer, Probability: 0.99009, box: (0.29941, 0.50782) (0.61335, 0.81000)
# Label: UnHealthy Deer, Probability: 0.02532, box: (0.34874, 0.00170) (0.58188, 0.55292)
# Label: Healthy Deer, Probability: 0.01254, box: (0.00925, 0.15533) (0.87207, 0.98420)

# (venv-python3113-datascience) C:\Development\GitHub\AIVisionExercises>python predictONNXS1OD.py 41c20ea0f5414a3381bfab173c1b97a2.ONNX\model.onnx TestImageUnHealthyDeerDayBuck.jpg
# Label: UnHealthy Deer, Probability: 0.13009, box: (0.07849, 0.59303) (0.71738, 0.99858)
# Label: Healthy Deer, Probability: 0.03130, box: (0.04158, 0.04641) (0.94205, 0.97782)
# Label: UnHealthy Deer, Probability: 0.01827, box: (0.55772, 0.04974) (0.78629, 0.40335)
# Label: UnHealthy Deer, Probability: 0.01671, box: (0.22403, 0.21533) (0.77413, 0.71315)

# (venv-python3113-datascience) C:\Development\GitHub\AIVisionExercises>
#
# 2023Oct02 mcvogt
# first an example of UnHealthy animal... 
# WORKING!!!!!! ========================== example ====================================
# |<-venv that was activated->| |<------- current directory -------->| python  |<-predict script->| |<----relative path from script to model->|    |<--image to be processed-->|
# (venv-python3113-datascience) C:\Development\GitHub\AIVisionExercises>python predictONNXS1OD.py 41c20ea0f5414a3381bfab173c1b97a2.ONNX/model.onnx TestImageUnHealthyDeerDayBuck.jpg
# Label: 1, Probability: 0.13009, box: (0.07849, 0.59303) (0.71738, 0.99858)
# Label: 0, Probability: 0.03130, box: (0.04158, 0.04641) (0.94205, 0.97782)
# Label: 1, Probability: 0.01827, box: (0.55772, 0.04974) (0.78629, 0.40335)
# Label: 1, Probability: 0.01671, box: (0.22403, 0.21533) (0.77413, 0.71315)

# (venv-python3113-datascience) C:\Development\GitHub\AIVisionExercises>

# second, an example of a Healthy animal...  
# (venv-python3113-datascience) C:\Development\GitHub\AIVisionExercises>python predictONNXS1OD.py 41c20ea0f5414a3381bfab173c1b97a2.ONNX/model.onnx TestImageHealthyDeerDayBuck.jpg
# Label: 0, Probability: 0.99009, box: (0.29941, 0.50782) (0.61335, 0.81000)
# Label: 1, Probability: 0.02532, box: (0.34874, 0.00170) (0.58188, 0.55292)
# Label: 0, Probability: 0.01254, box: (0.00925, 0.15533) (0.87207, 0.98420)

# (venv-python3113-datascience) C:\Development\GitHub\AIVisionExercises>

# labels to be read in from labels.txt
# 1 deer-day-healthy                    # index 0
# 2 deer-day-unhealthy                  # index 1

# 2023Sep11 mike evaluating approaches
# from https://github.com/Azure-Samples/customvision-export-samples/blob/main/samples/python/onnx/object_detection_s1/predict.py

# How to use...  python predict.py <model_filepath> <image_filepath>   <----   NO actual examples...   mike is frustrated.  
#  "C:\Development\GitHub\AIVisionExercises\41c20ea0f5414a3381bfab173c1b97a2.ONNX\model.onnx"
# python predictONNXS1OD.py 41c20ea0f5414a3381bfab173c1b97a2.ONNX/model.onnx TestImageHealthyDeerDayBuck.jpg
#========================================= example ====================================
#======================================================================================

# IMPORTS
import argparse
import pathlib
import numpy as np
import onnx
import onnxruntime
import PIL.Image

PROB_THRESHOLD = 0.01  # Minimum probably to show results.


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

    def predict(self, image_filepath):
        image = PIL.Image.open(image_filepath).resize(self.input_shape)
        input_array = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
        input_array = input_array.transpose((0, 3, 1, 2))  # => (N, C, H, W)
        if self.is_bgr:
            input_array = input_array[:, (2, 1, 0), :, :]
        if not self.is_range255:
            input_array = input_array / 255  # => Pixel values should be in range [0, 1]

        outputs = self.session.run(self.output_names, {self.input_name: input_array.astype(self.input_type)})
        return {name: outputs[i] for i, name in enumerate(self.output_names)}

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_filepath', type=pathlib.Path)
    parser.add_argument('image_filepath', type=pathlib.Path)

    args = parser.parse_args()

    model = Model(args.model_filepath)
    outputs = model.predict(args.image_filepath)
    print_outputs(outputs)


if __name__ == '__main__':
    main()
    
#======================================================================================================================================
#======================================================================================================================================
#======================================================================================================================================
    