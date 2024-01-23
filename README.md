  # AIMi10_for_AMLStudio
This is John Berk's first entry, setting up empty repo in GitHub first, then will clone to AML Studio and populate, commit, and sync back.  the notes for this repo will be stored and maintained in stack fashion (below this initial text) with oldest comments at the bottom and newest comments toward the top.   this project focuses on deploying an Azure Custom Vision Object Detection model called AIMi10, from the Azure CUV portal (customvision.ai), exported as a .onnx (ONNX) model, and making it available for on-line inferencing from an Azure Managed Endpoint.    

The actual model consists of four (4) files within the *Users/john/AIMi10_for_AMLStudio/TCAI_AIM_Iteration_10_on_Gen_Compact_S1.ONNX* folder.   these files include:   

*cvexport.manifest  
labels.txt  
LICENSE  
metadata_properties.json  
model.onnx*

*model.onnx* is used for all the inferencing, and label.txt contains the ordered category labels associated with each of the model's output indexes.    

*score021c.py* is the key single file that is uploaded to the Endpoint during Deployment, and used by Azure Machine Learning to finish building out a Docker Container that is the Endpoint host.  much of the code within *score021c.py* is taken directly from the stand-alone, command-line-callable *predict.py* (and *predictfromURL.py*) script which can be used to interface with and inference from the model.onnx when no web/cloud service is needed.   the key differences are how the two scripts receive input data from the end user (as elements in a HTTP REST POST request or as params in a command line).  

*consume021c.ipynb* is a Jupyter notebook file whos cells are edited to construct testable input sent to the AIMi10 Endpoint.  only the final two cells need concern the user.  these allow single-shot test files to be sent to the Endpoint, returning JSON dictionary output of the inferenced input.  the final cell launches a repeatedly looping test load at the service.   it also returns JSON dictionary output containing Bounding_Boxes, Probabilities, and Classes for the inferenced data.     

internal to the cell code it references an Azure Storage Account Blob Container that allows public access.  This contains the (30) or so test images that are used.   

*TestImageDeerBuckNight.jpg* and *TestImageDeerDoeDay.jpg* are a couple of local .jpg files to be used for local testing of the *predict.py* script.
  


2024/01/16 jberk   john@berkcg.com  
this is an overhaul update for this entire repo. this solution has been up and running since 2023/11/25.  this GitHub repo has been fully tested and cleaned up and represents the latest fully functional version of that original SOW#1 project.

the entire repo contains a .ONNX folder, (3) .py files, (1) .ipynb file, and a couple of .jpg files.  there is an associated Azure Storage Account set up with public access to test images.  



2024/01/04 jberk  
John-Michael launched VSCode integrated terminal, and from there found that the newly made Git repo had been moved to the User/john account on the *compute-ds11v2-aivisionT* host...   but the AML Workspace editor and session were linked from that user's home direction to ~/cloudfiles/code/   so in the terminal, JM used the cp -r command to copy the full set of AIMi10_for_AMLStudio files TO the /cloudfiles/code/Users/john/AIMi10_for_AMLStudio   and voila, they showed up Where They Should have in AML Studio file system...   but when JM went to rm -r the original source, he got messages about not being the owner, and them being associated w Git...  deleted anyway... will try and sync the new copy...

2024Jan23 jberk
John checking on some weirdness from VSCode running as IDE for Azure Machine Learning - test commit and up-sync
