import time
import argparse
import numpy as np
import robotic as ry
import matplotlib.pyplot as plt
from typing import Union, List, Any

# ===== Own packages =====
from camera.real_sense import RealSense
from object_detector.aruco_detector import detect_aruco_markers


def gen_scene() -> ry.Config:
    """
    Set up a scene with a Panda and 2 boxes.
    """
    C = ry.Config()

    C.addFile(ry.raiPath("scenarios/pandaSingle.g"))
    # Adding a camera and markers
    C.addFile("../config/config_markers_2.g")
    
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

def check_cfg(C: ry.Config, to_del: List[Any] = [], message="") -> None:
    user_key = C.view(pause=True, message="Press q to quit (any other key to continue)")

    if user_key == ord("q"):
        print("Received q. Exiting scene...")
        del C
        for obj in to_del:
            del obj
        exit(0)

    else:
        print("Continuing...")

def main(args):
    print("[Args: ", args, " ]")

    # Global parameters that can influence the friction, grasp force, etc.
    ry.params_add(
        {
            "physx/angularDamping": 0.1,
            "physx/defaultFriction": 100.0,
            "physx/defaultRestitution": 0.7,
            "botsim/verbose": 0,
        }
    )  # Increase verbosity to 2 for more info

    C = gen_scene()
    if not check_cfg(C):
        exit(0)

    cam = RealSense()
    cameras = {"top": "cameraTopMiddle", "wrist": "cameraWrist"}

    if args.display and args.input:
        bot = ry.BotOp(C=C, useRealRobot=args.real)
        if args.input in cameras:
            rgb, depth, _ = bot.getImageDepthPcl(cameras[args.input])     
        else:                                   # (args.input == "standalone")
            cam.connect()
            cam.start_grabbing()
            rgb, depth = cam.acquire_rgb_and_depth()
                           
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
        
        if args.input in cameras:                   
            bot = ry.BotOp(C=C, useRealRobot=args.real)
            rgb, _, _ = bot.getImageDepthPcl(cameras[args.input])  
            fxycxy = bot.getCameraFxycxy(cameras[args.input]) 
            
            fx, fy, cx, cy = fxycxy
            K = np.array([[fx, 0,  cx],
                        [0,  fy, cy],
                        [0,  0,  1.0]])
            
            del bot    

        else:                                   # args.input == "standalone"
            cam.connect()
            cam.start_grabbing()
            rgb, _ = cam.acquire_rgb_and_depth()
            K, dist, W, H = cam.get_color_intrinsics()           

        # Detect and display markers         
        detect_aruco_markers(rgb, K, dist, W, H) 


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--real", action="store_true", default=False, help="Use this arg if real robot is used")  # Use this arg to run on the real robot    
    p.add_argument("--input", type=str, default="wrist", help="Select camera: wrist, top or standalone")
    p.add_argument("--display", action="store_true", default=False, help="Show camera view")
    p.add_argument("--aruco", action="store_true", default=False, help="Detect AruCo markers in the RGB image")
    p.add_argument("--segment", action="store_true", default=False, help="Run instance segmentation to locate an object")

    args = p.parse_args()

    main(args)
