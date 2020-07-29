# This File runs the streamLit visualization for our model

import streamlit as st
import numpy as np
import json
import random
import cv2
import torch
import os
import json
from PIL import Image
# Detectron2 imports

import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import sys
from appMain import *

from Ensemble2 import *
from imports import *

# TODO Way to load model with @st.cache so it doesn't take a long time each time
#@st.cache(allow_output_mutation=True)

def main():
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.sidebar.image(Image.open("images/aryeo-banner.png"), width=250)
    st.title("Aryeo's Computer Vision Platform")
    st.write("This platform supports [Aryeo's](https://medium.com/airbnb-engineering/amenity-detection-and-beyond-new-frontiers-of-computer-vision-at-airbnb-144a4441b72e) automatic amenity detection")
    st.sidebar.markdown("## How does  it work?")
    st.sidebar.markdown("Add an image of a room and a machine learning learning model will look at it and find the amenities like the example below:")
    st.sidebar.image(Image.open("images/banner.png"), width=300)
   
    st.write("## Step1: Upload your own image")
    st.write("**Note:** Please choose a png or jpg file.")
    uploaded_image = st.file_uploader("Upload below", 
                                  type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
              
        
        #### found online code######## checks to see if image needs to be rotated
        
        try:
    
            if hasattr(image, '_getexif'): # only present in JPEGs
                for orientation in ExifTags.TAGS.keys(): 
                    if ExifTags.TAGS[orientation]=='Orientation':
                        break 
                e = image._getexif()       # returns None if no EXIF data
                if e is not None:
                    exif=dict(e.items())
                    orientation = exif[orientation] 
    
                    if orientation == 3:   image = image.transpose(Image.ROTATE_180)
                    elif orientation == 6: image = image.transpose(Image.ROTATE_270)
                    elif orientation == 8: image = image.transpose(Image.ROTATE_90)

        except:
              traceback.print_exc()
        
        image = image.convert("RGB")

        imageLocation = st.empty()
        imageLocation.image(image, caption="... Successfully Uploaded!", use_column_width=True)
        
        
        
        #if st.button("Rotate Image To Correct Orientation"):
           # image  = image.rotate(90)
           # imageLocation.image(image)

        
        
        
        
        st.write("## Step2: Detect Amenity")
        # TODO: Fix image size if the instance is breaking
        # If image is over certain size
        # Resize image to certain size
        # Don't make the image too small 
        #st.write(image.size)
        
        if st.button("Make a prediction"):
          # TODO: Add progress/spinning wheel here
          "Making a prediction and drawing bounding boxes on your image..."
          with st.spinner("Predicting..."):
              preDic = run(image)
              if not preDic:
                  st.image(image, caption = "no amenities detected")
              
              else:
                  st.pyplot(preDic["inputImage"]["output image"])
              
            
            
              #st.image(custom_pred, caption="Amenities detected.", use_column_width=True)

if __name__ == "__main__":
    main()
