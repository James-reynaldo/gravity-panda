
# Exercise 2: Pose Estimation with ArUco Markers

Goal: given an RGB(-D) camera and printed ArUco markers, estimate each marker’s pose in world coordinates. 
You’ll implement the missing steps for the pipeline **marker detection → marker pose in camera → marker pose in world coordinates**, and validate the results in a simple scene.

---

## Preliminaries 
### ArUco markers

ArUco markers are square, high-contrast fiducial markers with a unique ID and a known square geometry. Because the side length is known (always in meters and needs to be measured after printing) and if we can detect the 4 image corners reliably, we can solve a PnP problem to recover the marker’s pose with respect to the camera. 
Typical uses are robot to world alignment, object localization, and pose estimation. You can generate markers in different sizes (4x4, 5x5 etc). The actual size is always bigger due to the black boarder: a 5x5 marker will have 7x7 squares. In this exercise we are using the ARUCO_ORIGINAL 5x5 markers.

---

### Coordinate frames and matrices

We’ll use the 4×4 homogeneous transform convention:

- `T_a_b` contains rotation matrix `R_a_b` and translation vector `t_a_b`: 
  `X_a = R_a_b · X_b + t_a_b` and 
  `T_a_b = [[R_a_b, t_a_b],[0 0 0 1]]`.

**Frames used**

- `c` or `cam` — camera frame (OpenCV coordinates: x right, **y down**, z forward).
- `m` or `marker` — marker local frame.
- `w` or `world` — world frame.
 
---

### Intrinsics vs. extrinsics

- **Intrinsics `(K, dist)`** map camera rays to pixels. From RealSense you can read them from the device; from the simulation cameras you can get `(fx, fy, cx, cy)` directly (different from real camera). 
- **Extrinsics** are poses between frames (world to camera or world to marker).

You’ll use intrinsics to compute the **camera pose** of each marker, then compose extrinsics to get **world poses**.

Note: 
For estimating ArUco marker poses OpenCV provides `estimatePoseSingleMarkers`, which is not available in newer versions. For newer versions you can use `solvePnP` to fit the transform using the marker's corners (correct order is TL, TR, BR, BL).

---

## Instructions

We provide the initial script to simplify the task. The main file is `tutorial_2.py` with options `display`, `aruco` and `segment`.
`display` is already implemented and acquires and displays images either from the robot cameras or from a standalone attached Realsense camera. The robot cameras are `cameraTopMiddle` and `cameraWrist`. Note: if botop is in the simulation mode, virtual cameras are used instead. With `input` you can specify which camera you want to use for `display` or `aruco` detection.

```python
python3 tutorial_3.py --help
 
#usage: tutorial_3.py [-h] [--real] [--input INPUT] [--display] [--aruco] [--segment]
#
# options:
#  -h, --help     show this help message and exit
#  --real         Use this arg if real robot is used
#  --input INPUT  Select camera: wrist, top or standalone
#  --display      Show camera view
#  --aruco        Detect AruCo markers in the RGB image
#  --segment      Run instance segmentation to locate an object
```

The main part you will be working on is `aruco`.

### 0: Install and Test
Install packages with `pip install -r requirements.txt`. 
`requirements.txt` lists the high-level packages you would need for the `aruco` part. 
`segment` requires additionally the `ultralytics` package and we recommend to install it later as it introduces many additional dependencies.

Once your environment is ready, test it with 
`python tutorial_3.py --input top --display`

You should see the robot scene first and then (after pressing any button) the window with the RGB and depth images. Close the window with `q`.

### 1.1: Detect ArUco markers and visualize them [3 points]

`detect_aruco_markers` is the main method for this part. You can start the detection using `python tutorial_3.py --input top --aruco` for the top camera (or changing to another source)
You will need to implement several steps/subproblems inside it:

a) Detect ArUco markers. You may use `cv2.aruco` detector, which returns the corners and ids. Output the list of detected markers (in terminal or on the image)
b) Calculate poses of the markers using either `estimatePoseSingleMarkers` or `solvePnP` methods in openCV.
c) Visualize the detected markers overlaying them on the RGB image with their id and coordinate frames. You can use openCV/aruco methods `drawDetectedMarkers` and `drawFrameAxes`.

### 1.2: Calculate the markers poses in world frame coordinates from a known camera position [4 points]

If you know `T_w_c` and you compute `T_c_m` (marker poses you calculated in 1.1), then:
```
T_w_m = T_w_c · T_c_m
```
Compose the matrix needed to transform the pose to the world coordinates and output the poses. If using the virtual top camera you can easily verify the poses of the markers comparing them to the X (absolute) poses set in `markers.g`

Note: pay attention to the coordinate axes (s. OpenCV Y-axis issue below) and the order of the transforms.

### 1.3: Calculate the transform when one marker pose is known [3 points]
When the camera position is not known or is less precise, we can use a known marker pose to compute or update it. Which in turn can be used to find the world poses of other markers.
You can use any of the markers as the starter, the script currently assumes marker 100 is known.

If marker `k` has a known world transform `T_w_mk` and you computed `T_c_mk` and `T_c_mi` (marker poses), first recover `T_w_c`:
```
T_mk_c = inv(T_c_mk)
T_w_c  = T_w_mk · T_mk_c
```
Then for any other marker `i`:
```
T_w_mi = T_w_c · T_c_mi
```

Again, output the calculated poses and compare them to the ground truth poses in `markers.g`.

### OpenCV Y‑axis issue
OpenCV’s camera frame convention follows **y down**, such that the resulting poses will be accurate in the X and Z axis, but flipped in Y. 
Insert a single **Y‑flip** in the camera frame when composing:
```
T_flipY = diag([1, -1, 1, 1])
T_c'_m = T_flipY · T_c_m          # marker -> (y‑up camera)
T_w_m  = T_w_c · T_c'_m
```

### 2: Scene understanding with instance segmentation (Optional 3 points)
To use instance segmentation you would need to install the `ultralytics` package, preferably in a separate environment (due to dependencies it might change your openCV and numpy versions)
`segment` part of the `main` method contains a snippet on loading the segmentation model and defining custom classes.
You need to extend it with the following steps:

a) Run prediction on acquired images from one of the cameras
b) For the found objects, calculate the centers of the segmentation masks. 
c) Calculate and output 3D positions of the centers using the aligned depth image (relative to the camera).

---

