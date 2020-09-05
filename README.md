# Aryeo-Amenity-Detection
See a Demo of the Project [here](https://www.youtube.com/watch?v=Xy2T6KPusdE&ab_channel=AIEngineering)

The goal of this project was to develop an object detection model that is capable of extracting amenity data from household images. We were employed by [Aryeo](https://www.aryeo.com/). Inspired by Airbnb's [medium article,](https://medium.com/airbnb-engineering/amenity-detection-and-beyond-new-frontiers-of-computer-vision-at-airbnb-144a4441b72e) we built an object detection model by utilizing Facebook AI's [Detectron2 library.](https://github.com/facebookresearch/detectron2) More specifically, we conducted transfer learning on a Retinanet model found in the Detectron2 model zoo library. See the descriptions of our workspace below to navigate the repository. Most of our work was conducted in a Google VM Instance, so take a look at that folder. Also, note that unfortunately due to proprietary information, we are not able to share the actual models themselves. They are also too large to include on Github.

## Parent Directory
The three files found in the parent directory are our initial project notebooks completed in Google Colab. This was essentially our playground before we started seriously training our model.
The Airbnb_Rep.ipynp notebook is our first training Notebook of a small model. The Airbnb_rep_Data_cleaning.ipynb notebook shows our data cleaning process of changing "Open Images" (An open-source dataset) style labels to fit with Detectron2. The Detectron2-Small-model_Tutorial.ipynb is the initial tutorial we followed to learn about how to train Detectron2 models on custom datasets. 

 
## GoogleVM Folder
This folder contains all of our training notebooks, data collection notebooks, and data cleaning scripts.

### Small Model
Contains the training notebook and image download notebook for our small model. It was trained on only one class of amenity data, the coffeemaker class. We wanted to ensure our data cleaning worked, and that loss decreased as we trained. We ensured that our workflow was effective.

### Medium Test Model
Contains the experimentation notebook, in which we tested 7 different pre-trained models to determine the one on which we would conduct transfer learning. We tested both Retinanet and R-CNN models. We found that the Retinanet_R_101_3x model was the best.

### bigDogModel
This folder contains the image download and training notebooks for all 30 classes of amenities. The Save_Predictions notebook is used to save predictions from a model into a CSV file that is used for inferencing. This directory also includes a visualization notebook which visualizes predictions that we have passed through our ensembling script... (see  later section on ensembling)


#### pseudoLabeling
Pseudo labeling is the use of high confidence predictions, obtained by passing an unlabeled dataset through the model, as labels for those images to then be used to retrain the model. It is a way of generating more labeled data to train the model and hopefully increase the accuracy.

This folder contains our script, threshold_predictions.py, which is used to make predictions on a set of unlabeled images, pass them through our ensembling script, and then threshold them, only keeping predictions that have confidence levels above our threshold (70%). The script then turns them into Detectron2 style labels, dumps it to a JSON to be registered with Detectron2, and then creates a new training folder of images. 


### Ensembling
Ensembling is the process of combining multiple predictions from different models on the same piece of test data to generate one, ideally more accurate prediction. This folder contains Ensemble.py, which takes the predictions CSV output of the save_Predictions notebook from the BigDogModel folder, cleans the predictions to the appropriate format, and passes those predictions to the Ensemble script from https://github.com/ZFTurbo/Weighted-Boxes-Fusion. You can simply pip install ensemble-boxes to utilize the script. 

### mAP_Calculations
As is standard within object detection, mean average precision (mAP) was our main inferencing metric. While we were able to simply use the cocoEvaluator's mean average precision calculator on a single model, to inference on our ensembled system, we needed to generate a different way of measuring mAP. This folder contains our custom script which reformats our ensembled predictions on our validation set, reformats the validation labels, and saves both to separate text files to be used by the mAP script found on https://github.com/Cartucho/mAP

### Deploy
This folder contains our final demo code. We utilized Streamlit to build a user interface to interact with our system. The file, app.py, creates the front end that interacts with appMain.py. This file is responsible for passing the input image through our system of models, and returning output data on the locations and classes of the items detected, as well as an output image with bounding boxes. 

## Loss Function Visualization Script
This Folder was created to visualize the training metadata, and more specifically, visualize the loss vs iterations chart. The purpose of this was to determine whether or not we needed to train our model for more iterations. Essentially we wanted to see if the loss had flattened out over time or if it was still decreasing (i.e. learning). The script in this folder is adaptable to anyone utilizing detectron2. Read the Instructions.txt file in that folder if you are interested.

