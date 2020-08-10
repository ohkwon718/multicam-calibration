# multicam-calibration

Optimize(refine) calibration of multiple camera. The camera parameters are obtained from 3D-2D matching and are further optimized from 2D points across cameras. This code is developed for [**Bonn Activity Maps**](https://cg.cs.uni-bonn.de/en/publications/paper-details/bonn_activity_maps_arxiv/).



![result](https://user-images.githubusercontent.com/19196641/89715651-6bd35800-d9a7-11ea-85e1-fb7fb6adb472.png)


### Requirements 

* python
* numpy
* opencv
* pandas

### Input Description
(input).json:
```
{
  "3d_points_file":"data/(3d_points).txt",
  "cameras":
  {
    "cam0": 
    {
      "img_file":"data/(cam0).jpg",
      "calib_file":"data/(cam0).json",
      "points_file":"data/(cam0).txt"
    },
    "cam1": 
    {
      "img_file":"data/(cam1).jpg",
      "calib_file":"data/(cam1).json",
      "points_file":"data/(cam1).txt"
    },
    ...
  }
}
```

(3d_points).txt:
```
target1 (float) (float) (float) \\ 3d position (x, y, z)
target2 (float) (float) (float) \\ 3d position (x, y, z)
...
```

(cam0).json:
```
{
  "name": "cam0", 
  "K": [[(float), (float), (float)], [(float), (float), (float)], [(float), (float), (float)]], \\ 3 x 3 intrinsic calibration parameter 
  "R": [(float), (float), (float)], \\ rotation vector (not required for optimization)
  "t": [(float), (float), (float)], \\ translation vector (not required for optimization)
  "imgSize": [(int), (int)] \\ image size (w, h)
}
```

(cam0).txt:
```
target1 (int) (int) \\ pixel (x, y)
target2 (int) (int) \\ pixel (x, y)
```
