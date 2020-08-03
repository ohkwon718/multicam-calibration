import os 
import json
import argparse
from collections import OrderedDict
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Calibrate multiple cameras.')
parser.add_argument('--json', type=str, default='data/input.json', help='json file defining all input files')
parser.add_argument('--path-res', type=str, default='result', help='folder to save result')
parser.add_argument('--iter', type=int, default=10, help='iteration to optimize')

args = parser.parse_args()

with open(args.json) as json_file:  
    data = json.load(json_file)

path = os.path.dirname(args.json)
df_3d_pts = pd.read_csv(os.path.join(path, data['3d_points_file']), sep = ' ', header=None, index_col=0)

cameras = data['cameras']
for key in cameras:
    camera = cameras[key]
    camera['img'] = cv2.imread(os.path.join(path, camera['img_file']))
    with open(os.path.join(path, camera['calib_file'])) as json_file:  
        camera['calib'] = json.load(json_file)
    camera['2d_pts'] = pd.read_csv(os.path.join(path, camera['points_file']), sep = ' ', header=None, index_col=0)
    camera['2d_pts_unknown'] = camera['2d_pts'].index.difference(df_3d_pts.index)
args.dist = True
flag = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6 + cv2.CALIB_ZERO_TANGENT_DIST 
dist = np.zeros((1,5))

def get_triangulation(xys, camera_matrixs):
    """
    param xys : list of points in cameras
    param camera_matrixs : np.array{ v x 3 x 4 }
    return X,s : the point in 3d space and corresponding singular value
    """
    A = np.empty((0,4))
    for x, camera_matrix in zip(xys, camera_matrixs):
        A = np.append(A, x[:2].reshape(2,1) @ camera_matrix[2:3] - camera_matrix[:2], axis = 0)
    
    A /= np.linalg.norm(A[:,:3], axis=-1, keepdims=True)
    xyz, res, _, _ =  np.linalg.lstsq(A[:,:3], -A[:,-1:], rcond=None)
    return xyz[:,0], res

df_3d_pts_all = df_3d_pts
for i in tqdm(range(args.iter)):
    camera_matrixs = dict()
    for key in cameras:
        idx_inter = cameras[key]['2d_pts'].index.intersection(df_3d_pts_all.index)
        object_points = df_3d_pts_all.loc[idx_inter, 1:].to_numpy(dtype=np.float32)
        image_points = cameras[key]['2d_pts'].loc[idx_inter, 1:].to_numpy(dtype=np.float32)
        mtx = cameras[key]['calib']['K'] = np.array(cameras[key]['calib']['K'])
        imageSize = tuple(cameras[key]['calib']['imgSize'])
        retval, mtx, dist, rvec, tvec = cv2.calibrateCamera([object_points], [image_points], imageSize, mtx, dist, flags=flag)
        cameras[key]['calib']['K'] = mtx
        cameras[key]['calib']['dist'] = dist
        cameras[key]['calib']['R'] = rvec[0]
        cameras[key]['calib']['t'] = tvec[0]
        camera_matrixs[key] = mtx @ np.hstack((cv2.Rodrigues(rvec[0])[0], tvec[0]))
        
    unknown_2d_pts = dict()
    for key in cameras:
        tgts = cameras[key]['2d_pts_unknown']
        for tgt in tgts:
            if tgt in unknown_2d_pts:
                unknown_2d_pts[tgt][key] = cameras[key]['2d_pts'].loc[tgt].to_numpy()
            else:
                unknown_2d_pts[tgt] = {key:cameras[key]['2d_pts'].loc[tgt].to_numpy()}

    idxs = []
    new_3d_pts = []
    for tgt in unknown_2d_pts:
        if len(unknown_2d_pts[tgt]) >= 2:
            xys = []
            cms = []
            for cam in unknown_2d_pts[tgt]:
                xys.append(unknown_2d_pts[tgt][cam])
                cms.append(camera_matrixs[cam])
            xyz, err = get_triangulation(xys, cms)
            new_3d_pts.append([tgt] + xyz.tolist())
    
    df_3d_pts_all = df_3d_pts.append(pd.DataFrame(new_3d_pts).set_index(0))


for key in cameras:
    calib = OrderedDict()
    calib['name'] = cameras[key]['calib']['name']
    calib['K'] = cameras[key]['calib']['K'].tolist()
    calib['distCoef'] = cameras[key]['calib']['distCoef']
    calib['R'] = cameras[key]['calib']['R'].reshape(-1).tolist()
    calib['t'] = cameras[key]['calib']['t'].reshape(-1).tolist()
    calib['imgSize'] = cameras[key]['calib']['imgSize']
    with open(os.path.join(args.path_res, cameras[key]['calib_file']), 'w') as outfile:  
        json.dump(calib, outfile, indent = 2)
