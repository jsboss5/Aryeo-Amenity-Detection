# This file will take predictions from ensemble.py and clean them to reuse as labels (sudo labeling). it will take the dictionary returned from ensemble.py, delete all images without a detection of .7 confidence or higher, delete those corresponding images from the test image folder, and create a detectron2 style label dictionary
import cv2
import json
import sys
sys.path.insert(1, '../../Ensembling')

from Ensemble import *

def threshold(dic, thresh):
    # Takes a dictionary dic, which is the output of the ensemble.py file (ensembled predictions)
    # Thresh - threshold for keeping a prediction - above this value will be used for sudo labels
    # Returns a new dictionary but only includes predicitons above threshhold variable
    
    retDic = {}         #create new dictionary that only good scores will be added to
    counter = 0
    for fileName in dic.keys():        #Loop through all of the files in the dictionary
      
        newBoxes = []                  # only boxes with scores of thresh or higher will be added
        newLabels = []                 # same with labels
        newScores = []                #same with scores
        
        for dex in range(len(dic[fileName]['scores'])):         #loop through the number of predictions
            score = dic[fileName]['scores'][dex]                #loop through scores for each prediciton
            if score>thresh:                                     # If a score for a prediction > threshold
                newBoxes.append(dic[fileName]['boxes'][dex])       # add that prediction to all the new lists
                newLabels.append(dic[fileName]['labels'][dex])
                newScores.append(dic[fileName]['scores'][dex])
        dic[fileName]['boxes'] = newBoxes                          #replace old lists with new lists
        dic[fileName]['scores'] = newScores
        dic[fileName]['labels'] = newLabels
        
        if len(dic[fileName]['scores']) == 0:                       #if the image has no predictions, print that it was delted
            x = "this is filler code"
        
        else:   
            retDic[fileName] = dic[fileName]                      # if there is at least one prediction add to the retDictionary
            #print(retDic[fileName])
        
    
    return retDic
    
def WandH(imagePath):
    img = cv2.imread(imagePath)
    height, width, channels = img.shape
    
    return height, width

def rel_to_abs(bbox, height, width):        #converts relative to absolute
    bbox[0] =  bbox[0] * width   #x0
    bbox[1] =  bbox[1] * height  #y0
    bbox[2] =  bbox[2] * width  #x1
    bbox[3] =  bbox[3] * height  #y1
    
    return  (bbox)    

def jsonDump(dic, outputFile):
    # This function creates Detectron2 Style labels from the thresholded output of Ensemble.py
    
    dumpList = []
    counter = 0
    for key in dic.keys():
        #if counter == 100:
         #   break
        d = {}
        path = '/'.join(key.split('/')[1:])
        height = WandH(path)[0]
        width = WandH(path)[1]
        
        #print(path)
        d["file_name"] = path
        d["image_id"] = counter
       
        d["height"] = height
        d["width"] = width
        
        d["annotations"] = []
        
        
        for dex in range(len(dic[key]['boxes'])):
            annotationDic = {}
            box = dic[key]["boxes"][dex]
            box = rel_to_abs(box, height, width)
            annotationDic["bbox"] = box
            annotationDic["bbox_mode"] = 0
            annotationDic["category_id"] = int(dic[key]["labels"][dex])
            d["annotations"].append(annotationDic)
            
        
        counter +=1
        dumpList.append(d)
        
       
    with open(outputFile, 'w') as outfile:      # dump the giant list to a json file to be registered
        json.dump(dumpList, outfile)
    
    return
     
def main():    
    inputCSV = "preEnsemble.csv"    #change this as needed
    outputJSON = 'sudoLabels.json'
    thresh = .7  # Change this to be whatever you want the threshold for label confidence to be
    
    ensembledDic = engine(inputCSV, outputJSON)         #stores dictionary ouput from ensemble.py
    thresholded_dic = threshold(ensembledDic, thresh)   #stores dictionary with only most accurate
    
    jsonDump(thresholded_dic, outputJSON)             #turn into detectron2 style and dump to a json
    
if __name__ == "__main__":
    main()