import os 
import json
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Calibrate multiple cameras.')
parser.add_argument('--json', type=str, default='data/input.json', help='json file defining all input files')

args = parser.parse_args()

with open(args.json) as json_file:  
    data = json.load(json_file)


