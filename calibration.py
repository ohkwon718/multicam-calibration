import os 
import json
import argparse
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Calibrate multiple cameras.')
parser.add_argument('--json', type=str, default='data/input.json', help='json file defining all input files')
parser.add_argument('--path-res', type=str, default='res', help='folder to save result')

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
    

flag = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6 + cv2.CALIB_ZERO_TANGENT_DIST 
dist = np.zeros((1,5))

for key in cameras:
    idx_inter = cameras[key]['2d_pts'].index.intersection(df_3d_pts.index)
    object_points = df_3d_pts.loc[idx_inter, 1:].to_numpy(dtype=np.float32)
    image_points = cameras[key]['2d_pts'].loc[idx_inter, 1:].to_numpy(dtype=np.float32)
    mtx = np.array(cameras[key]['calib']['K'])
    imageSize = tuple(cameras[key]['calib']['imgSize'])
    print(cameras[key]['calib'])
    retval, mtx, dist, rvec, tvec = cv2.calibrateCamera([object_points], [image_points], imageSize, mtx, dist, flags=flag)
    cameras[key]['R'] = rvec[0]
    cameras[key]['t'] = tvec[0]

