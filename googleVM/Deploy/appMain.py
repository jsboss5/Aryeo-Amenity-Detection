'''
ALL NECESSARY IMPORTED PACKAGES CAN BE FOUND IN 'otherPyScripts/imports.py'
'''


'''
THIS FILE IS WHAT RUNS THE PREDICTIONS ON A SINGLE IMAGE... UTILIZES ENSEMBLE TECHNIQUES COMBINING THE BEST PREDICTIONS OF 4 DIFFERENT MODELS WE TRAINED

RETURNS A DICTIONARY THAT INCLUDES INFORMATION ABOUT BOUNDING BOX OF PREDICTIONS, CONFIDENCES, AND A PYPLOT THAT CAN BE VISUALIZED BY STREAMLIT FOR THE USER INTERFACE
'''
import sys
sys.path.insert(1, 'otherPyScripts')
from Ensemble2 import *
from imports import *
from app import main as app_py_Main

import streamlit as st

# Global Variables
subset = ['Bathtub', 'Bed', 'Billiard table', 'Ceiling fan', 'Coffeemaker', 'Couch', 'Countertop', 'Dishwasher', 'Fireplace', 'Fountain', 'Gas stove', 'Jacuzzi', 'Kitchen & dining room table', 'Microwave oven', 'Mirror', 'Oven', 'Pillow', 'Porch', 'Refrigerator', 'Shower', 'Sink', 'Sofa bed', 'Stairs', 'Swimming pool', 'Television', 'Toilet', 'Towel', 'Tree house', 'Washing machine', 'Wine rack']

'''
CODE STARTS
'''


def makePredictors(configsDir):
    '''
    Makes a list of predictors from config files in the modelConfigs folder
    '''
    
    predList = [] 
    
    for filename in os.listdir(configsDir):
        if filename.endswith(".yaml"):
            cfg = get_cfg()
            cfg.merge_from_file(configsDir + filename)
            
            predictor = DefaultPredictor(cfg) 
            
            predList.append(predictor)
            
    
    return predList


def absolute_to_rel(bbox, height, width):
    bbox[0] =  bbox[0] / width   #x0
    bbox[1] =  bbox[1] / height  #y0
    bbox[2] =  bbox[2] / width  #x1
    bbox[3] =  bbox[3] / height  #y1
    
    return  (bbox)


def predict(predList, inputImage):    #This was taken from save predictions nb
    inputImage
    d = {}
    
    imgID = "inputImage"
    d["id"] = imgID
    d["boxes"] = []
    d["scores"] = []
    d["classes"]= []
    
    shape = inputImage.size     #gets a tuple (height, width)
    width = shape[0]     #sets height
    height = shape[1]      #sets width variable
    
    for predictor in predList:
        inputImage = np.asarray(inputImage)
        x = predictor((inputImage))
        tens = x['instances']
        numInstances = tens.scores.size()[0]
        Boxes = tens.pred_boxes
        Boxes = (Boxes.tensor)
        Boxes = Boxes.cpu()
        Boxes = Boxes.numpy()   #Boxes in numpy array
    
    
        scores = tens.scores
        scores = scores.cpu().numpy()    #scores in numpy array
    
        classes = tens.pred_classes.cpu().numpy()  #classes in numpy array
    
        Boxes = Boxes.tolist()            #boxes is now a list of lis
        scores = scores.tolist()          #now a list
        classes = classes.tolist()
    


        for Box in Boxes:            
            Box = absolute_to_rel(Box, height, width)
        
        d["boxes"].append(Boxes)
        d["scores"].append(scores)
        d["classes"].append(classes)

    return d
    


def ensembleEngine(preDictionary):  
    '''
    Calls function from Ensemble2.py
    '''
    
    iou_thr = .5                    #set variables
    skip_box_thr = .0001
    sigma = 0.1
    method = "weighted_boxes_fusion" 
          
        
    #from the ensemble2 script
    
    return imgDetect(preDictionary, iou_thr, skip_box_thr, sigma, method)  

def rel_to_matplotlib(bbox, img):
# THIS FUNCTION WILL TURN THE RELATIVE COORDINATES TO THE MATPLOT FORM (X1,Y1, WDITH, HEIGHT)
    #img = cv2.imread(img_path)
    IMwidth, IMheight = img.size    #gets a tuple (height, width)
    
        
    x0 = bbox[0]* IMwidth        #THIS GETS THE FIRST X COORDINATE
    y0 = bbox[1] * IMheight       #THIS GETS BOTTOM LEFT Y COORDINATE
    
    absX = IMwidth * bbox[2]
    absY = IMheight * bbox[3]
    

    width = absX - x0
    height = absY - y0
    
    return [x0,y0,width, height]


def visualize(dic, im):
    
    inputImage = im
    
    for key in dic:
        #image = mpimg.imread(imagePath)
        #plt.imshow(image)
        #plt.show
        
        
        path = "inputImage"
        bboxes = dic[key]["boxes"]
        
       
        
        im = np.array(im)
        
        fig, ax = plt.subplots(1, figsize = (8,8))
        ax.imshow(im)
        
        right = ax.spines["right"]
        right.set_visible(False)
        left = ax.spines["left"]
        left.set_visible(False)
        top = ax.spines["top"]
        top.set_visible(False)
        bottom = ax.spines["bottom"]
        bottom.set_visible(False)
        
        
        plt.hist([1], orientation=u'vertical')
        plt.xticks([])
        plt.yticks([])
        
        dex = 0
        for i in range(len(bboxes)):
            bbox = bboxes[i]
            label = dic[key]["labels"][i]
            label = subset[int(label)]
                        
            color = np.random.rand(3,)
            newbbox = rel_to_matplotlib(bbox, inputImage)
            rect = patches.Rectangle((newbbox[0],newbbox[1]), newbbox[2], newbbox[3], linewidth=3,edgecolor=color,facecolor='none')
            ax.add_patch(rect)
            ax.annotate(label + " "+ str(dic[key]["scores"][dex]), color = color, xy = (newbbox[0],newbbox[1]))
           
            dex+=1
    
    #plt.savefig('foo.png')   You can save this is if you want to see the output
    
    
    
    return plt


def make_final_dic(ensembledDic, figure):
    for img in ensembledDic.keys():
        for i in range(len(ensembledDic[img]["labels"])):
            ensembledDic[img]['labels'][i] = subset[int(ensembledDic[img]['labels'][i])]    # converts the float number to actual class
        ensembledDic[img]["output image"] = figure
    
    return ensembledDic
        

    
    
def engine(configsDir, imagePath, inputImage):
    
    predList = makePredictors(configsDir)     # make a list of predictions
    
    preDictionary = (predict(predList, inputImage)) #returns dictionary of predicts 
    ensembledDic = (ensembleEngine(preDictionary))  #returns post ensemble dictionary - new boxes and scores
    
    plt = visualize(ensembledDic, inputImage)        # Visualize these predictions and store them in plt
    
    finalDic =  (make_final_dic(ensembledDic, plt))
    
   
            
    
    return finalDic
    
    
    
def run(inputImage):    
   
    
    # This should include only variables that can be changed and call engine
    configsDir = "modelConfigs/"
    imagePath = "testImages/couch.jpeg"    # This might be like a list of images
    return engine(configsDir, imagePath, inputImage)
    

    
if __name__ == "__main__":
    print(run("lol"))

