import json
import csv
import sys
sys.path.insert(1, "../../Ensembling")
import Ensemble
import numpy as np
import ast
import cv2
# TODO
# 1. Change teh functions to not need to add small model or bigdogmodel
    
    
# true_label_to_text and ensemble_prediction_to_text should return the same structure of dictionary, except that the ensemble prediction to text utilizes the scores

#An example of a dictionary entry is this - (for the prediction)
#'validation/ff170cdebcd8d899.jpg': {'boxes': [[0.0, 0.0, 561.99524050951, 1014.5580444335938]], 'labels': ['Coffeemaker'], 'scores': [0.9068222641944885]}

def true_label_to_dic(filename, subset):
    #Returns a dictionary with file path as key and dictionary as value where the dictionary value is
    # has two keys, boxes and labels, parralel lists where one represents bounding box & other labels
    #Example
        # {'validation/ff170cdebcd8d899.jpg': {'boxes': [[0.0, 0.0, 590.1729736328125, 984.8470458984375]], 'labels': ['Coffeemaker']}
    
    retDic = {}
    with open(filename) as json_file:
        data = json.load(json_file)
        for p in data:                             #p is a dictionary
            d = {}
            d["boxes"] = []
            d["labels"] = []
            for annotation in (p["annotations"]):           
                d["boxes"].append(annotation['bbox'])            
                d["labels"].append(subset[annotation['category_id']])   #subset is the list of classes 
            
            retDic[p["file_name"]] = d
               
        return retDic
            

def ensemble_prediction_to_dic(filename, subset):
    #The filename is a JSON file that is one big dictionary where keys are file paths (starting with dfasdfkjf.jpg - not validation/asdfer.jpg) and the key is another dictionary which is a list of bounding boxes, scores and labels all parallel list    -> see "../../Ensembling/smallModelEnsembleJSON.json" for an example
     
    
    with open(filename) as json_file:
        data = json.load(json_file)
        retDic = {}                         #declare the return dictionary
        for p in data.keys():               #loop through the keys
            path = "validation/" + p          #create the new path by adding the jpeg to the validation
            img = cv2.imread("../../BigDogModel/" + path)    #read the image to attain heaight and width
            shape = img.shape
            height = shape[0]    #sets height
            width = shape[1]
            
            d = {}                            #create the dictionary that will be value in retDic
           
            d["boxes"] = []                    #loop through and change the coordinates to absolute
            for box in data[p]["boxes"]:
                d["boxes"].append(rel_to_abs(box, height, width))     #append to the dictionary
              
            
            
            d["labels"] = []                             #Change the number label to the human class
            for label in data[p]["labels"]:
                d["labels"].append(subset[int(label)])
            
            d["scores"] = data[p]["scores"]              #just copy the scores list

            
            retDic[path] = d
        
        return(retDic) 
        
def rel_to_abs(bbox, height, width):        #converts relative to absolute
    bbox[0] =  bbox[0] * width   #x0
    bbox[1] =  bbox[1] * height  #y0
    bbox[2] =  bbox[2] * width  #x1
    bbox[3] =  bbox[3] * height  #y1
    
    return  (bbox)    
    
def normal_prediction_to_dic(filename, subset):
    
    #Input file is a csv with the following format
    # id,boxes,scores,classes
    #ff170cdebcd8d899.jpg,"[[[0.0, 0.0, 0.833527645060521, 0.9830200672149658]]]",[[0.9848663210868835]],[[0]]

    #This function should create the same dictionary as the above two but from a csv file which has id, boxes, scores, classes as the columns... 
    
    with open(filename) as csvfile:                            #read the CSV file
        readCSV = csv.reader(csvfile, delimiter = ',')
        next(readCSV)
        retDic = {}
        for row in readCSV:                                  #Skip the title portion of csv
            d = {}
            path = "validation/" + row[0]                     #update the path to include validation
            
            img = cv2.imread("../../BigDogModel/" + path)    #read the image to attain heaight and width
            shape = img.shape
            height = shape[0]                         #sets height
            width = shape[1]                           #sets width
            
# Deal with boxes 
            
            boxesString = row[1]                        #CSV OBJECTS IS A STRING
            boxes_list = ast.literal_eval(boxesString)     #TURN THE sTRING INTO A LIST
            boxes_list = boxes_list[0]        #Because adapted from ensemble,there extra set of brackets
            for i in range(len(boxes_list)):
                boxes_list[i] = rel_to_abs(boxes_list[i], height, width)  #Convert coordinates to abs

# Deal with Scores            
            
            scores_string = row[2]                    
            scores_list = ast.literal_eval(scores_string)[0]
#Deal with labels            
            labels_string = row[3] 
            labels_list = ast.literal_eval(labels_string)[0]
            for dex in range(len(labels_list)):
                labels_list[dex] = subset[labels_list[dex]]
          
            d["boxes"] = boxes_list              #add all lists to value dictionary
            d["labels"] = labels_list
            d["scores"] = scores_list

            retDic[path] = d                      #add to ret dictionary
       
    return retDic


def write_dic_to_text(d, trueDic, outputDir, gtORpr, ):
    
    
    for key in trueDic.keys():
        title = key.split('/')[1].split('.')[0]       #just the image id aka ff170casde
        textTitle = title + '.txt'
        
        file = open(outputDir + textTitle, 'w')
        # I need to wrap this in if statement that makes sure its in dictionary
        
        if key in d:       #should check to see if the key is in the dictionary , if not write blank 
    
            for i in range(len(d[key]["boxes"])):
                file.write(d[key]["labels"][i] + " ")

                if gtORpr == "pr":
                    file.write(str(d[key]["scores"][i]) + " ")

                for coord in d[key]["boxes"][i]:
                    file.write(str(int(coord)) + " ") 
                file.write('\n')

        file.close()
           
    
    
    #TODO
    return
    
    
if __name__ == "__main__":
   
    # These are the variables for small model    
    
    #subset = ["Coffeemaker"]    Put the subset in alphabetical ordr
    #True_Labels = "../../smallModel/validation/validation_labels.json"    #Put the True Labels
    #ensemble_labels = "../../Ensembling/smallModelEnsembleJSON.json"
    #csvPredictions = "singlePredictionSM.csv"  
    
  

   # These are variables for BDM
    
    subset = ["Toilet",
          "Swimming_pool",
          "Bed",
          "Billiard_table",
          "Sink",
          "Fountain",
          "Oven",
          "Ceiling_fan",
          "Television",
          "Microwave_oven",
          "Gas_stove",
          "Refrigerator",
          "Kitchen_&_dining_room_table",
          "Washing_machine",
          "Bathtub",
          "Stairs",
          "Fireplace",
          "Pillow",
          "Mirror",
          "Shower",
          "Couch",
          "Countertop",
          "Coffeemaker",
          "Dishwasher",
          "Sofa_bed",
          "Tree_house",
          "Towel",
          "Porch",
          "Wine_rack",
          "Jacuzzi"]

    subset.sort()

    True_Labels = "../../BigDogModel/validation/validation_labels.json"    #Change this to true label
    ensemble_labels = "../../Ensembling/BigDogModelEnsembleJSON.json"      #Change this to ensemble
    csvPredictions = "singlePredictionBDM.csv"                        #change this to single predict
    
    true_dic = true_label_to_dic(True_Labels, subset)
    ensemble_dic = ensemble_prediction_to_dic(ensemble_labels, subset)
    predic_dic = normal_prediction_to_dic(csvPredictions, subset)
    
    gtORpr = "gt"                                # ground truth
    outputDir = "input/ground-truth/"            # output Directory
    #write_dic_to_text(true_dic, true_dic, outputDir, gtORpr)     #write the text files for ground truth
    
    gtORpr = "pr"
    outputDir = "input/detection-results/"
   
    write_dic_to_text(ensemble_dic, true_dic, outputDir, gtORpr)  #write the text files for predictions, here it is ensemble

    #ALso remember to change the functions to reflect the big or small Dog model