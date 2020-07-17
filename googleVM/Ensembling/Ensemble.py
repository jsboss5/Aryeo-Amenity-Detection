import csv
from ensemble_boxes import *
import numpy as np
import ast

#RIGHT NOW IT GENERATES A DICTIONARY WHERE THE KEY IS THE IMAGE ID "232387234.JPG" AND THE VALUE IS
#A DICTIONARY WITH KEYS BOXES, SCORES, LABELS...
#IT WORKS RIGHT NOW, EXCEPT I MIGHT HAVE TO CHANGE OUTPUT ARRAYS FROM NPY TO NORMAL


def ensemble(iou_thr, skip_box_thr, sigma, method, weights, boxes, scores, labels):   
    
    #Calls the correct Ensemble method
    if method == "nms":
        boxes, scores, labels = nms(boxes, scores, labels, weights=weights,iou_thr=iou_thr)
    elif method == "soft_nms":
        boxes, scores, labels = soft_nms(boxes, scores, labels, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)
    elif method == "non_maximum_weighted":
        boxes, scores, labels = non_maximum_weighted(boxes, scores, labels, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    elif method == "weighted_boxes_fusion":
        boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    
    return [boxes, scores, labels]




def imgDetect(filename, iou_thr, skip_box_thr, sigma, method):
    #return list of 
    
    boxes_list = [] #Get from some json or CSV
    scores_list = [] #Get from the json or CSV
    labels_list = [] #Get from the json or CSV
                     #IDK make it up
    
    retDic = {}     #Create dictionary that we will return
    
    
    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter = ',')    #read the csv
        next(readCSV)                       # Skip the header
        for row in readCSV:                 # for each row 
            imgID = row[0]
            boxes_list = row[1]         #make boxes_list = first part of each line
            scores_list = row[2]        #make scores_list second part of line
            labels_list = row[3]        #make labels_list 3rd part of line

      # This converts all these strings to lists to be passed to the funtions below                    
            boxes_list = ast.literal_eval(boxes_list)
            scores_list = ast.literal_eval(scores_list)
            labels_list = ast.literal_eval(labels_list)
            
       # THIS FIRST CHECKS IF THERE ARE NO PREDICTIONS AT ALL, AND CONTINUES IF SO      
            empty = True
            for dex in range(len(scores_list)): #if everything is empty continue
                if not not scores_list[dex]:   #if theres something there
                    empty = False
                    
            if empty == True:   # IF THERE ARE NO PREDICTIONS AT ALL 
                continue        # CONTINUE TO NEXT LINE
                                 
            for i in range(len(scores_list)):  #Loop to check EACH prediction empty
                if not scores_list[i]:
                    del scores_list[i]  #if there is no prediction = delete  
                    del boxes_list[i]   #if there is no prediction = delete
                    del labels_list[i]  #if there is no prediction = delete
                    
                
            if not (scores_list):    # check AGAIN after deletions 
                continue             #to see if there are no predictions at all
            
           
        
        # Adjust number of weights based on number of non empty predictions
        
        weights = []                        
            for i in range(len(scores_list)):   # Loop through
                weights.append(1)               # append a 1 to weights for len
            
          
        #Creates a dictionary to add as value to key for each image
           
            valueDic = {}     # Create a dictionary that will b value of id key
            valueDic["boxes"] = ensemble(iou_thr, skip_box_thr, sigma, method, weights, boxes_list, scores_list, labels_list)[0]
            
            valueDic["scores"] = ensemble(iou_thr, skip_box_thr, sigma, method, weights, boxes_list, scores_list, labels_list)[1]
            valueDic["labels"] = ensemble(iou_thr, skip_box_thr, sigma, method, weights, boxes_list, scores_list, labels_list)[2]
            
            retDic[imgID] = valueDic    #make value dic the value of the id key
            #break
            
    return retDic
    

    
if __name__ == "__main__":    
    iou_thr = 0.5
    skip_box_thr = .0001
    sigma = 0.1
    method = "nms"      # change this to get different types of ensembling
    filename = "../smallModel/predictions_output.csv"    #change this as needed
    print(imgDetect(filename, iou_thr, skip_box_thr, sigma, method))