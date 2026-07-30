import os
import json
import argparse
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

def parse_map(fname):
    with open(fname, 'r') as fd:
        gt_dict = json.load(fd)
    aruco_dict = {}
    object_dict = {}

    for key in gt_dict:
        if key.startswith('aruco'): # read SLAM map
            aruco_num = int(key.strip('aruco')[:-2])
            aruco_dict[aruco_num] = np.reshape([gt_dict[key]['x'], gt_dict[key]['y']], (2, 1))
        else: # read object map
            object_type = key.split('_')[0]
            object_dict[object_type] = np.reshape([gt_dict[key]['x'], gt_dict[key]['y']], (2, 1))
    return aruco_dict, object_dict

def match_dict_key(d1, d2):
    # pair up the values from the same keys
    points1 = []
    points2 = []
    keys = []
    for key in d1:
        if not key in d2:
            continue
        points1.append(d1[key])
        points2.append(d2[key])
        keys.append(key) 
    return keys, np.hstack(points1), np.hstack(points2)

def compute_rmse(points1, points2):
    assert (points1.shape[0] == 2)
    assert (points1.shape[0] == points2.shape[0])
    assert (points1.shape[1] == points2.shape[1])
    num_points = points1.shape[1]
    residual = (points1 - points2).ravel()
    MSE = 1.0 / num_points * np.sum(residual ** 2)
    return np.sqrt(MSE)
    
def eval_slam(aruco_est, aruco_gt):
    taglist, slam_est_vec, slam_gt_vec = match_dict_key(aruco_est, aruco_gt)
    theta, x = solve_umeyama2d(slam_est_vec, slam_gt_vec)
    slam_est_vec_aligned = apply_transform(theta, x, slam_est_vec)
    diff = slam_gt_vec - slam_est_vec_aligned
    slam_rmse_raw = compute_rmse(slam_est_vec, slam_gt_vec)
    slam_rmse_aligned = compute_rmse(slam_est_vec_aligned, slam_gt_vec)

    # print()
    # print("The following parameters optimally transform the estimated points to the ground truth.")
    # print("Rotation Angle: {}".format(theta))
    # print("Translation Vector: ({}, {})".format(x[0,0], x[1,0]))
    print()
    print("Number of found markers: {}".format(len(taglist)))
    print(f'SLAM RMSE before alignment = {np.round(slam_rmse_raw, 5)}')
    print(f'SLAM RMSE after alignment = {np.round(slam_rmse_aligned, 5)}')
    print()
    print('%s %7s %9s %7s %11s %9s %7s' % ('Marker', 'Real x', 'Pred x', 'Δx', 'Real y', 'Pred y', 'Δy'))
    print('-----------------------------------------------------------------')
    for i in range(len(taglist)):
        print('%3d %9.2f %9.2f %9.2f %9.2f %9.2f %9.2f\n' % (taglist[i], slam_gt_vec[0][i], slam_est_vec_aligned[0][i], diff[0][i], slam_gt_vec[1][i], slam_est_vec_aligned[1][i], diff[1][i]))
    
    ax = plt.gca()
    ax.scatter(slam_gt_vec[0,:], slam_gt_vec[1,:], marker='o', color='C0', s=100)
    ax.scatter(slam_est_vec_aligned[0,:], slam_est_vec_aligned[1,:], marker='x', color='C1', s=100)
    for i in range(len(taglist)):
        ax.text(slam_gt_vec[0,i]+0.05, slam_gt_vec[1,i]+0.05, taglist[i], color='C0', size=12)
        ax.text(slam_est_vec_aligned[0,i]+0.05, slam_est_vec_aligned[1,i]+0.05, taglist[i], color='C1', size=12)
    plt.title('Arena')
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_xticks([-1.5, -1, -0.5, 0, 0.5, 1.0, 1.5])
    ax.set_yticks([-1.5, -1, -0.5, 0, 0.5, 1.0, 1.5])
    plt.axis([-1.6, 1.6, -1.6, 1.6])
    plt.legend(['Real','Pred'])
    plt.show()
    return slam_rmse_aligned, (theta, x)

def solve_umeyama2d(points1, points2):
    # Solve for optimal transform theta and x such that theta*p1 + x = p2

    assert (points1.shape[0] == 2)
    assert (points1.shape[0] == points2.shape[0])
    assert (points1.shape[1] == points2.shape[1])

    num_points = points1.shape[1]
    mu1 = 1 / num_points * np.reshape(np.sum(points1, axis=1), (2, -1))
    mu2 = 1 / num_points * np.reshape(np.sum(points2, axis=1), (2, -1))
    Sig12 = 1 / num_points * (points2 - mu2) @ (points1 - mu1).T

    # Use the SVD for the rotation
    U, d, Vh = np.linalg.svd(Sig12)
    S = np.eye(2)
    if np.linalg.det(Sig12) < 0:
        S[-1, -1] = -1

    # Return the result as an angle and a 2x1 vector
    R = U @ S @ Vh
    theta = np.arctan2(R[1, 0], R[0, 0])
    x = mu2 - R @ mu1
    return theta, x

def apply_transform(theta, x, points):
    assert (points.shape[0] == 2)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    points_transformed = R @ points + x
    return points_transformed
 

def eval_object(object_est, object_gt, transform=None):
    # returns the error (euclidean distance) for each individual object estimation against gt
    # calculate two error versions: before and after alignment, and obtain the smaller of the two
    
    MAX_ERROR = 1
    full_obj_list = list(object_gt.keys())
    errors = {}
    # initialize error dictionary with max error
    for obj_name in full_obj_list:
        errors[obj_name] = MAX_ERROR
    
    # detection result
    obj_list, obj_est_vec, obj_gt_vec = match_dict_key(object_est, object_gt)
    for i, obj_name in enumerate(obj_list):
        err = np.linalg.norm(obj_est_vec[:,i] - obj_gt_vec[:,i])
        errors[obj_name] = np.round(err, 5)
    avg_error = sum(errors.values()) / len(errors)
    if transform is None: print('Note: When evaluating object pose only, transform is not applied.')
    
    # if need to apply transform, calculate error after transform
    if transform is not None:
        theta, x = transform
        object_est_vec_aligned = apply_transform(theta, x, obj_est_vec)
        errors_after = {}
        for i, obj_name in enumerate(obj_list):
            err = np.linalg.norm(object_est_vec_aligned[:,i] - obj_gt_vec[:,i])
            errors_after[obj_name] = min(np.round(err, 5), errors[obj_name])
        avg_error_after = sum(errors_after.values()) / len(errors_after)
        if avg_error_after < avg_error:
            avg_error = avg_error_after
            errors = errors_after
    
    print('Object pose estimation errors:')
    print(json.dumps(errors, indent=4))
    # print(f'Average object pose estimation error: {sum(errors.values()) / len(errors)}')
    return errors

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Matching the estimated map and the true map')
    parser.add_argument('--truemap', type=str, default='truemap.txt')
    parser.add_argument('--slam-est', type=str, default='lab_output/slam.txt')
    parser.add_argument('--object-est', type=str, default='lab_output/objects.txt')
    args, _ = parser.parse_known_args()

    aruco_gt, object_gt = parse_map(args.truemap)
    
    slam_only, object_only = False, False
    if os.path.exists(args.slam_est):
        aruco_est, _ = parse_map(args.slam_est)
        if len(aruco_est) == 0 or len(aruco_gt) == 0:
            object_only = True
    else:
        object_only = True
    
    if os.path.exists(args.object_est):
        _, object_est = parse_map(args.object_est)
        if len(object_est) == 0 or len(object_gt) == 0:
            slam_only = True
    else:
        slam_only = True


    if slam_only: # only evaluate SLAM
        print('Evaluating SLAM only:')
        slam_rmse_aligned, _ = eval_slam(aruco_est, aruco_gt)
    elif object_only: # only evaluate object
        print('Evaluating Object Detection only:')
        object_est_errors = eval_object(object_est, object_gt, transform=None)
    else: # evaluate both
        print('Evaluating both SLAM & Object Detection:')
        slam_rmse_aligned, transform = eval_slam(aruco_est, aruco_gt)
        object_est_errors = eval_object(object_est, object_gt, transform=transform)