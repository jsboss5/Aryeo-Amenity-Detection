# Aryeo-Amenity-Detection
The goal of this project is to be able to extract amenity data from household images for our employer Aryeo.com. Inspired by AirBnb's [medium article](https://medium.com/airbnb-engineering/amenity-detection-and-beyond-new-frontiers-of-computer-vision-at-airbnb-144a4441b72e) We built an object detection model utilizing  

## Parent Directory
We contain all of our work done in google colab (Before we set up a google cloud virtual machine) 
The Airbnb_Rep.ipynp Notebook is a our first training Notebook of a small model.The Airbnb_rep_Data_cleaning.ipynb shows data cleaning process of changing OpenImages labels to Detectron2 labels. The Detectron2-Small-model_Tutorial.ipynb is the tutiral we followed to learn about how to use Detectron2. 

 
## GoogleVM Folder
This Folder contains all of our Training Notebooks, Data collection notebooks, and data cleaning Scripts.

### Small Model
Contains the training notebook and image download NB for our small model, trained on one class of image

### Medium Test Model
Contains the experimentaton notebook, in which we tested 7 different pretrained models to determine the one we would conduct transfer learning on.

### bigDogModel
This folder contains the image download and training notebooks similar for all 30 classes of amenities. The Save_Predictions notebook is used to save predictions from a model into a csv file that is used for inferencing. It also includes a visualization notebook which visualizes predictions that we have passed through our ensembling script... (see next later section on Ensembling)


#### pseudoLabeling
Pseudo labeling is the use high confidence predictions on an unlabeled dataset as labels for those images to then be used to retrain the model. It is a way of generating more labeled data

This folder contains our script which, threshold_predictions.py, which is used to take our predictions on a set of unlabled images, pass them through our ensembling script, and then threshold them, only keeping predictions that have confidence levels above our threshold (70%). The script then turns them into Detectron2 style labels, writes it to a Json to be registered with Detectron 2, and then creates a new training folder of images. 


### Ensembling
This folder contains Ensemble.py, which takes the predictions csv output of the save_Predictions notebook from the BigDogModel folder, cleans the predictions to the appropriate format and passes those predictions to the Ensemble script from https://github.com/ZFTurbo/Weighted-Boxes-Fusion. You can simply pip install ensemble-boxes to utilize the script. 

### mAP_Calculations
Our way of inferencing on the model. Mean average precision calculation. This folder contains our custom script which reformats our ensembled predictions on our validation set, and correct labels for the validation set to the appropriate form, and saves them to seperate text files to be used by the mAP script found on https://github.com/Cartucho/mAP

### Deploy
This folder contains our final demo code. We utilized streamlit to build a user interface to interact with our system. app.py creates the front end and then pulls from appMain.py which actually passes the input image through our system and returns an output image. 

