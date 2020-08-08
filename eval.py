import os 
import json
import argparse
from collections import OrderedDict
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Calibrate multiple cameras.')
parser.add_argument('--json', type=str, default='data/input.json', help='json file defining all input files')
parser.add_argument('--path-res', type=str, default='result/figures', help='folder to save result')
args = parser.parse_args()

os.makedirs(args.path_res, exist_ok=True)
with open(args.json) as json_file:  
    data = json.load(json_file)

df_3d_pts = pd.read_csv(data['3d_points_file'], sep = ' ', header=None, index_col=0)

cameras = data['cameras']
for key in cameras:
    camera = cameras[key]
    camera['img'] = cv2.imread(camera['img_file'])
    with open(camera['calib_file']) as json_file:  
        camera['calib'] = json.load(json_file)
    camera['2d_pts'] = pd.read_csv(camera['points_file'], sep = ' ', header=None, index_col=0)
    camera['2d_pts_unknown'] = camera['2d_pts'].index.difference(df_3d_pts.index)


dist = np.zeros((1,5))
camera_matrixs = {}
camera_centers = {}
for key in cameras:
    idx_inter = cameras[key]['2d_pts'].index.intersection(df_3d_pts.index)
    object_points = df_3d_pts.loc[idx_inter, 1:].to_numpy(dtype=np.float32)
    image_points = cameras[key]['2d_pts'].loc[idx_inter, 1:].to_numpy(dtype=np.float32)
    mtx = np.array(cameras[key]['calib']['K'])
    rvec = np.array(cameras[key]['calib']['R'])
    tvec = np.array(cameras[key]['calib']['t'])
    camera_matrixs[key] = mtx @ np.hstack((cv2.Rodrigues(rvec)[0], tvec[:,None]))
    camera_centers[key] =  -np.linalg.inv(camera_matrixs[key][:,:3]) @ camera_matrixs[key][:,3]


def get_cross(A):
    return np.array([[0, -A[2], A[1]], [A[2], 0, -A[0]], [-A[1], A[0], 0]])

epipoles = {}
fundamental_matrixs = {}
for c1 in cameras:
    for c2 in cameras:
        if c1 == c2:
            continue
        epipole = camera_matrixs[c2] @ np.append(camera_centers[c1], 1)
        epipole = epipole/epipole[2] 
        fundamental_matrixs[c1, c2] = get_cross(epipole) @ camera_matrixs[c2] @ np.linalg.pinv(camera_matrixs[c1])
        epipoles[c2, c1] = epipole[:2]


unknown_2d_pts = dict()
for key in cameras:
    tgts = cameras[key]['2d_pts_unknown']
    for tgt in tgts:
        if tgt in unknown_2d_pts:
            unknown_2d_pts[tgt][key] = cameras[key]['2d_pts'].loc[tgt].to_numpy()
        else:
            unknown_2d_pts[tgt] = {key:cameras[key]['2d_pts'].loc[tgt].to_numpy()}

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

colors = {}
np.random.seed(0)
for key in df_3d_pts_all.index:
    if key in df_3d_pts.index:
        colors[key] = np.random.randint(256, size=3)
    else:
        colors[key] = np.zeros(3)
print('camera\terr1\terr2')
for c1 in cameras:
    img = cameras[c1]['img']
    errs_predefined = []
    errs_new = []
    for tgt in cameras[c1]['2d_pts'].index:
        if tgt not in df_3d_pts_all.index:
            continue
        color = colors[tgt]
        cv2.circle(img, tuple(cameras[c1]['2d_pts'].loc[tgt].tolist()), 5, color.tolist(), 4)
        if tgt in df_3d_pts_all.index:
            pt = camera_matrixs[c1] @ np.append(df_3d_pts_all.loc[tgt], 1)
            pt = (pt/pt[-1])[:-1]
            err = (np.linalg.norm(pt - cameras[c1]['2d_pts'].loc[tgt]))
            if tgt in df_3d_pts.index:
                errs_predefined.append(err)
            else:
                errs_new.append(err)
    print(c1, '{:.3f}\t{:.3f}'.format(np.array(errs_predefined).mean(), np.array(errs_predefined+errs_new).mean()))

    for c2 in cameras:
        if c1 == c2:
            continue
        cv2.circle(img, tuple(epipoles[c1,c2].astype(int).tolist()), 5, (150,150,150),3)
    
        for tgt in cameras[c2]['2d_pts'].index:
            if tgt not in cameras[c1]['2d_pts'].index:
                continue
            color = colors[tgt]
            l = np.append(cameras[c2]['2d_pts'].loc[tgt],1) @ fundamental_matrixs[c1,c2]
            w1 = int(epipoles[c1,c2][0])
            
            w2 = cameras[c1]['2d_pts'].loc[tgt, 1] 
            w2 = int((l[1]**2*cameras[c1]['2d_pts'].loc[tgt, 1] - l[0]*l[1]*cameras[c1]['2d_pts'].loc[tgt, 2] - l[0]*l[2])/(l[0]**2 + l[1]**2))
            w2 = w2 - 1 if cameras[c1]['2d_pts'].loc[tgt, 1] < w1 else w2 + 1
            cv2.line(img, (w1, int(-(l[0]*w1+l[2])/l[1])), (w2 , int(-(l[0]*w2+l[2])/l[1])), color.tolist(), 1)
            
    cv2.imwrite(os.path.join(args.path_res, c1+'.png'), img)
