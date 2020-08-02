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
    
# df_3d_pts.loc[''].to_numpy()
