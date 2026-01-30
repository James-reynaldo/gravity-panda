import argparse
import robotic as ry
import numpy as np
from typing import Tuple, List, Any
from pprint import pprint
from cv2 import aruco
import cv2 as cv
import pyrealsense2 as rs
import matplotlib.pyplot as plt

# Optional: segmentation model
# from ultralytics import YOLOE

class DetectedObject():
    def __init__(self, name :str=None, bbox :np.ndarray=np.array([None, None, None, None])):
        self.name :str = name
        self.bbox: np.ndarray = bbox
        self.center: list = self.get_bbox_center(self.bbox.tolist())
        self.q_pose = [None, None, None, None, None, None, None]

    def get_bbox_center(self, bbox) -> list:
        if bbox is None or any(v is None for v in bbox):
            return [None, None]
        
        x_min, y_min, x_max, y_max = bbox
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        return [x_center, y_center]
    
    def print_obj(self):
        print()
        print(f"{self.name.capitalize()}")
        print(f"- BBox: {self.bbox}")
        print(f"- Center: {self.center}")
        print(f"- Pose (quat.): {self.q_pose}")


### Realsense Utils ###

def start_realsense(serial=None, width=1280, height=720, fps=30):
    pipeline = rs.pipeline()
    config = rs.config()
    if serial:
        config.enable_device(serial)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipeline.start(config)
    # Set up alignment so depth matches color frame
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    return pipeline, profile

def acquire_images_realsense(serial=None):
    pipeline, _ = start_realsense(serial=serial)
    try:
        while True:        
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
         
            if not color_frame:
                continue
            return color_frame, depth_frame                     
    finally:
        pipeline.stop()

    return None, None
    
def get_color_intrinsics(profile):
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    # intr: width, height, ppx, ppy, fx, fy, model, coeffs (5)
    K = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0, 0, 1]], dtype=np.float64)
    # Map RS distortion to OpenCV k1,k2,p1,p2,k3 (Brown-Conrady)
    if intr.model in (rs.distortion.brown_conrady, rs.distortion.inverse_brown_conrady):
        coeffs = np.array(intr.coeffs[:5], dtype=np.float64)
    else:
        coeffs = np.zeros(5, dtype=np.float64)
    return K, coeffs, intr.width, intr.height

def rt_to_T(R, t):
    T = np.eye(4, dtype=float)
    T[:3,:3] = R
    T[:3, 3] = t.reshape(3)
    return T

def rvec_tvec_to_T(rvec, tvec):
    R, _ = cv.Rodrigues(rvec)
    return rt_to_T(R, tvec)     
  
def matrix2vector(T):
   vec = np.zeros(7) 
   R = T[:3, :3]
   quat = ry.Quaternion().setMatrix(R).asArr() 
   vec[:3] = T[:3, 3]
   vec[3:] = quat
   return vec           
            
def detect_aruco_markers(rgb, K, dist, W, H):
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
    params = aruco.DetectorParameters_create()

    out = rgb.copy()
    detection_txt = "Poses (cam):"
    # T_world2cam = np.array([[-1, 0, 0, 0],[0, 1, 0, 0.45],[0, 0, -1, 2.30],[0, 0, 0, 1]])       #world coordinates of the virtual top down camera. Will not work for other inputs!
    T_world2marker30 = np.array([[1, 0, 0, 0],[0, 1, 0, 0.3],[0, 0, 1, 0.65],[0, 0, 0, 1]])         #world coordinates of marker 100
    T_cam2marker100 = np.eye(4)
    marker_length = 0.095  # 0.05 would be standard for academic projects

    #### BEGIN SOLUTION for 1.1

    object_points = np.array([
        [-marker_length/2,  marker_length/2, 0],
        [ marker_length/2,  marker_length/2, 0],
        [ marker_length/2, -marker_length/2, 0],
        [-marker_length/2, -marker_length/2, 0]
    ], dtype=np.float32)

    # --- Detect markers using old-style API ---
    marker_corners, marker_ids, rejected_candidates = aruco.detectMarkers(
        out,
        dictionary,
        parameters=params
    )

    
    if marker_corners is None or marker_ids is None:
        print("No markers detected. Exit program.")
        return
    
    # Build marker dict
    marker_dict = {int(m_id[0]): {"xyxy": corners} for m_id, corners in zip(marker_ids, marker_corners)}


    
    # Compute and draw poses  
    for marker_id, marker_data in marker_dict.items():
        corners = marker_data.get('xyxy')
        retval,rvec,tvec = cv.solvePnP(objectPoints=object_points, imagePoints=corners, cameraMatrix=K, distCoeffs=dist)
        if retval:
            marker_dict.get(marker_id)["pnp"] = (rvec,tvec)
            cv.drawFrameAxes(out, K, dist, rvec, tvec, 0.05)

    # Draw marker bboxes 
    aruco.drawDetectedMarkers(out, marker_corners, marker_ids)
    cv.putText(out, detection_txt, (3,10), cv.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,255), 1, cv.LINE_AA)
    cv.imshow("ArUco Markers", out)

    # T_flipY = np.diag([1, -1, 1, 1])
    # m30_data= marker_dict.get(30)
    # T_m30_c = np.linalg.inv(m30_data.get('T_c_m_w2c'))
    


    
    ### 1.2: 
    ### Steps: convert rvec, tvec from 1.1 to a homogeneous transformation matrix using rvec_tvec_to_T
    ### Apply the flip transformation (s. README)
    ### Compute the world pose for the marker

    T_flipY = np.diag([1, -1, 1, 1])

    # print("==============================================================")
    # print("Using 'T_world2cam' to compute marker world coordinates:")
    # print()
    for marker_id, marker_data in marker_dict.items():
        if marker_id == 30:
            rvec, tvec  = marker_data.get('pnp')
            T_c_m = rvec_tvec_to_T(rvec, tvec)
            T_c_m_flip = T_flipY @ T_c_m
            T_world2cam = T_world2marker30 @ np.linalg.inv(T_c_m_flip)
            T_world2cam2= T_world2marker30 @ np.linalg.inv(T_c_m)
            # T_w_m  = T_world2cam @ T_c_m_flip
            # pose = matrix2vector(T_w_m)

            # marker_data['T_c_m_w2c'] = T_c_m
            # marker_data['T_w_m_w2c'] =  T_w_m
            # marker_data['pose_w2c'] = pose

            # print(f"Marker {marker_id} Pose: {np.round(pose, 2)}")
            # print()
            print(T_world2cam)
            print(T_world2cam2)
            print(T_c_m)
            print(T_c_m_flip)


                        
    # ### 1.3: 
    # ### Repeat with using T_world2marker100 instead of the camera pose 
    
    # m100_data = marker_dict.get(100)
    # T_m100_c = np.linalg.inv(m100_data.get('T_c_m_w2c'))
    # T_cam2marker100 = T_world2marker100 @ T_m100_c

    # print("==============================================================")
    # print("T_w_c (camera pose): ", T_cam2marker100)
    # print()
    # print("==============================================================")
    # print("Using 'T_world2marker100' to compute marker world coordinates:")
    # print()
    # for marker_id, marker_data in marker_dict.items():

    #     T_c_m = marker_data.get('T_c_m_w2c')
    #     T_c_m_flip = T_flipY @ T_c_m
    #     T_w_m  = T_cam2marker100 @ T_c_m_flip
    #     pose = matrix2vector(T_w_m)

    #     marker_data['T_c_m_w2m'] = T_c_m
    #     marker_data['T_w_m_w2m'] =  T_w_m
    #     marker_data['pose_w2m'] = pose
    #     print(f"Marker {marker_id} Pose: {np.round(pose, 2)}")
    #     print()                         

    cv.putText(out, detection_txt, (3,10), cv.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,255), 1, cv.LINE_AA)
    cv.imshow("ArUco Markers", out)
    cv.waitKey(-1)
    cv.destroyAllWindows()                
            
def gen_scene(filename: str = "panda_scene.g") -> ry.Config:
    """
    Set up a scene with a Panda and 2 boxes.
    """
    C = ry.Config()

    C.addFile(ry.raiPath("scenarios/pandaSingle.g"))
    # Adding a camera and markers
    C.addFile("./configs/markers.g")

    world = C.getFrame("world")
    world.setShape(ry.ST.marker,0.1)
    
    box1_pos_init = [-0.25, 0.1, 0.675]
    box2_pos_init = [0.25, 0.1, 0.675]

    # Add first box
    f = (
        C.addFrame("box1") 
        .setPosition(box1_pos_init)
        .setShape(ry.ST.ssBox, size=[0.05, 0.05, 0.05, 0.005])
        .setColor([1.0, 1.0, 0.0])
        .setMass(0.1)
        .setContact(1)
    )

    # Axes for first box
    f = C.addFrame(name="box1_axes", parent="box1")
    f.setShape(type=ry.ST.marker, size=[0.05])

    # Add second box
    f = (
        C.addFrame("box2")
        .setPosition(box2_pos_init)
        .setShape(ry.ST.ssBox, size=[0.05, 0.05, 0.05, 0.005])
        .setColor([0.0, 1.0, 1.0])
        .setMass(0.1)
        .setContact(1)
    )

    # Axes for second box
    f = C.addFrame(name="box2_axes", parent="box2")
    f.setShape(type=ry.ST.marker, size=[0.05])      

    # Create a frame 5 cm above placebox
    # This should be used to position the EE for placing (after picking)
    # f = C.addFrame('placeframe', f"box{args.placebox}").setRelativePosition([0.,0.,0.05]).setShape(type=ry.ST.marker, size=[0.03])    

    return C
    

def check(C: ry.Config, to_del: List[Any] = [], message="") -> None:
    user_key = C.view(pause=True, message="Press q to quit (any other key to continue)")

    if user_key == ord("q"):
        print("Quitting")

        del C
        for o in to_del:
            del o

        exit(0)
    else:
        print("Continuing")

def acquire_rgb_and_depth():
    color_frame, depth_frame = acquire_images_realsense()
    rgb = np.asanyarray(color_frame.get_data())
    depth = np.asanyarray(depth_frame.get_data())  
    return rgb, depth   

def process_yolo_result(results):
    detection_list = []
    for result in results:
            for box in result.boxes:
                name = f"{result.names[int(box.cls[0])]}"
                bbox = box.xyxy[0]
                detection_list.append(DetectedObject(name, bbox))
    return detection_list

def main(args):
    
    # Global parameters that can influence the friction, grasp force, etc.
    ry.params_add(
        {
            "physx/angularDamping": 0.1,
            "physx/defaultFriction": 100.0,
            "physx/defaultRestitution": 0.7,
            "botsim/verbose": 0,
        }
    )  # Increase verbosity to 2 for more info

    # Set up the scene
    C = gen_scene()
    check(C)
    
    cameras = {"top": "cameraTopMiddle", "wrist": "cameraWrist"}
    
    if args.display and args.input is not None :
        bot = ry.BotOp(C=C, useRealRobot=args.real)
        if args.input == "wrist" or args.input == "top":
            rgb, depth, _ = bot.getImageDepthPcl(cameras[args.input])       
        if args.input == "standalone":
            rgb, depth = acquire_rgb_and_depth()
                           
        fig = plt.figure(figsize=(10,5))
        axs = fig.subplots(1, 2)
        axs[0].imshow(rgb)
        axs[1].matshow(depth)    
        plt.show()  
        
    if args.aruco:
        W = 1280
        H = 720
        dist = np.zeros(5)
        bot = ry.BotOp(C=C, useRealRobot=args.real)
        
        if args.input == "wrist" or args.input == "top":
            rgb, _, _ = bot.getImageDepthPcl(cameras[args.input])       
        if args.input == "standalone":
            rgb, _ = acquire_rgb_and_depth()            
        
        if args.input == "wrist" or args.input == "top":                    
            rgb, _, _ = bot.getImageDepthPcl(cameras[args.input])  
            fxycxy = bot.getCameraFxycxy(cameras[args.input]) 
            
            fx, fy, cx, cy = fxycxy
            K = np.array([[fx, 0,  cx],
                          [0,  fy, cy],
                          [0,  0,  1.0]])
            
            del bot

        if args.input == "standalone":           
            pipeline, profile = start_realsense()
            K, dist, W, H = get_color_intrinsics(profile)           
            
        #detect and display markers         
        detect_aruco_markers(rgb, K, dist, W, H)                                             
        
        
        
## 2.2 (Optional): perform instance segmentation to locate objects
    if args.segment:
        try:
            from ultralytics import YOLOE
        except Exception as e:
            raise SystemExit("ultralytics is required. Install with: pip install ultralytics") from e
    
        # Initialize a YOLOE: Real-Time Seeing Anything
        model = YOLOE("yoloe-11l-seg.pt")  # or smaller models yoloe-11s/m-seg.pt
        # Set text prompts (open vocabulary)
        names = ["yellow cube", "blue cube", "ball"]
        model.set_classes(names, model.get_text_pe(names))
           
        # TO ADD: Acquire images
                         
        bot = ry.BotOp(C=C, useRealRobot=args.real)
            
        fxycxy = bot.getCameraFxycxy(cameras[args.input]) 
        
        fx, fy, cx, cy = fxycxy

        K = np.array([[fx, 0,  cx],
                        [0,  fy, cy],
                        [0,  0,  1.0]])
        
        
        
        # if args.input == "wrist" or args.input == "top":       
        #     rgb, _, _ = bot.getImageDepthPcl(cameras[args.input]) 
        
        # elif args.input == "standalone":           
        #     pipeline , profile = start_realsense()
        #     K, dist, W, H = get_color_intrinsics(profile)           
        #     rgb, _ = acquire_rgb_and_depth() 

        # else:  
        rgb = "cubes.png"
        
            
    
        # Run segmentation on acquired images                         
        results = model.predict(rgb)    #Image for testing prediction
        
        # Display the masks and labels    
        results[0].show()  

        # TO ADD: Compute centers of the segmentation masks
        detected_objects :list[DetectedObject] = process_yolo_result(results)


        # ADD: Calculate the 3D positions (in the camera frame) of the centers 

        dist = np.zeros(5)

        T_world2cam = np.array([[-1, 0, 0, 0],[0, 1, 0, 0.45],[0, 0, -1, 2.30],[0, 0, 0, 1]])
        T_flipY = np.diag([1, -1, 1, 1]) 
        marker_length = 0.092  # 0.05 would be standard for academic projects

        object_points = np.array([
            [-marker_length/2,  marker_length/2, 0],
            [ marker_length/2,  marker_length/2, 0],
            [ marker_length/2, -marker_length/2, 0],
            [-marker_length/2, -marker_length/2, 0]
        ], dtype=np.float32)

        for obj in detected_objects:

            xmin, ymin, xmax, ymax = obj.bbox
            image_points = np.array([
                [xmin, ymin],   # top-left
                [xmax, ymin],   # top-right
                [xmax, ymax],   # bottom-right
                [xmin, ymax]    # bottom-left
            ], dtype=np.float32)

            retval,rvec,tvec = cv.solvePnP(objectPoints=object_points, imagePoints=image_points , cameraMatrix=K, distCoeffs=dist)  
            if retval:
                T_c_m = rvec_tvec_to_T(rvec, tvec)
                T_c_m_flip = T_flipY @ T_c_m
                T_w_m  = T_world2cam @ T_c_m_flip
                q_pose = matrix2vector(T_w_m)
                obj.q_pose = q_pose
                obj.print_obj()
    else:
        del C
        exit(0)

if __name__ == "__main__":

    p = argparse.ArgumentParser()
    p.add_argument("--real", action="store_true", default=False, help="Use this arg if real robot is used")  # Use this arg to run on the real robot    
    p.add_argument("--input", type=str, default="wrist", help="Select camera: wrist, top or standalone")
    p.add_argument("--display", action="store_true", default=False, help="Show camera view")
    p.add_argument("--aruco", action="store_true", default=False, help="Detect AruCo markers in the RGB image")
    p.add_argument("--segment", action="store_true", default=False, help="Run instance segmentation to locate an object")

    args = p.parse_args()

    print(args)

    main(args)
