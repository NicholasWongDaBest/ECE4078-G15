# estimate the pose of a detected object
import os
import ast
import json
import csv
import numpy as np

    
# estimate the pose (x,y) of a detected object given its bbox and the robot's pose
def estimate_pose(robot_pose, box, object_true_height, focal_length):
    
    ######### Replace with your codes #########
    # TODO: compute pose of the object based on bounding box [x,y,width,height] and robot's pose [[x],[y],[theta]]
    # You may want to use the true height of the object and the focal length also
    # This is the default code which estimates every pose to be (0,0)
    
    pose_x, pose_y = 0.0, 0.0
    
    ###########################################
    
    return pose_x, pose_y
    

# merge the estimations of the objects so that there is only 1 final estimate for each object type
def merge_estimations(object_pose_dict):  
    """
    Input object dict format:
    object_pose_dict = {'obj1': [(x1,y1), (x2,y2), ...], 'obj2': [(x1,y1), (x2,y2), ...], ...}
    
    Return:
    object_pose_dict_final: {'obj1_0': {'x': ??, 'y': ??}, 'obj2_0': {'x': ??, 'y': ??}, ...}
    The '_0' is added for code compatibility purpose with other scripts.
    """
    
    ######### Replace with your codes #########
    # TODO: the operation below is the default solution, which simply takes the first estimation for each object type.
    # Replace it with a better merge solution.
    object_pose_dict_final = {}
    for key in object_pose_dict:
        try:
            first_estimate = object_pose_dict[key][0]
            pose_x = first_estimate[0]
            pose_y = first_estimate[1]
            object_pose_dict_final[key + '_0'] = {'x': pose_x, 'y': pose_y}
        except:
            object_pose_dict_final[key + '_0'] = {'x': 0.0, 'y': 0.0}
    ###########################################
    
    return object_pose_dict_final


if __name__ == "__main__":
    
    '''
    # Every line is pred.txt is: one robot pose (when the image is taken) + one or more bounding boxes of the detected object(s) (visualized in pred_X.png)
    # For every box (every detected object), you need to calculate an estimate of the object pose (x,y)
    # After estimating obj poses for all images (all lines in pred.txt), each object will have more than one estimates (because each obj may appear in more than 1 image)
    # You need to merge the estimates so that every object has only one final estimate
    '''
    
    # load camera matrix parameter which is required for object pose estimation
    fileK = "{}intrinsic.txt".format('./calibration/param/')
    camera_matrix = np.loadtxt(fileK, delimiter=',') 
    focal_length = camera_matrix[0][0]
    
    # load the list of object names and dimensions
    with open('object_list.csv', 'r') as file:
        reader = csv.DictReader(file)
        data = list(reader)
    object_list = [row['object'] for row in data]
    object_dimensions = {}
    for row in data:  
        object_dimensions[str(row['object'])] = [float(row['length(m)']), float(row['width(m)']), float(row['height(m)'])]
    
    # Initialize the object pose prediction as a dictionary (object name as key, list of estimate poses as value)
    object_pose_dict = {}
    for object_name in object_list:
        object_pose_dict[object_name] = []
    
    # Compute estimates
    with open('lab_output/pred.txt') as fp:
        
        # for every line in pred.txt (every image taken)
        for line in fp.readlines():
            entry = ast.literal_eval(line)
            robotpose, bboxes = entry['robotpose'], entry['bboxes']
            
            # for every bounding box detected
            for bbox in bboxes:
                predicted_class = bbox[0]
                box = bbox[1]
                true_height = object_dimensions[predicted_class][2]
                object_pose_dict[predicted_class] = estimate_pose(robotpose, box, true_height, focal_length)

    # merge the estimations of the objects so that there are only one estimate for each object type
    object_pose_dict = merge_estimations(object_pose_dict)
                     
    # save object pose estimations
    with open('lab_output/objects.txt', 'w') as fo:
        json.dump(object_pose_dict, fo, indent=4)
    
    print('Estimations saved!')