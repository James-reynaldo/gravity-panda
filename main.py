import time
import argparse
import numpy as np
import robotic as ry
import matplotlib.pyplot as plt
from typing import Union, List, Any

# ===== Own packages =====
from camera.real_sense import RealSense
from object_detector import detector as det
from simulation.sim_wrapper import SimWrapper
from motion import komo_planner as planner


def gen_scene():
    """Generate robot scene (environment) and return robotic-Config."""
    C = ry.Config()
    C.addFile(ry.raiPath("scenarios/pandaSingle.g"))
    C.delFrame("panda_collCameraWrist")
    C.addFile("config/config_gp.g")
    
    return C

def check_config(C: ry.Config, to_del: List[Any] = [], message="") -> bool:
    user_key = C.view(pause=True, message="Press q to quit (any other key to continue)")
    if user_key == ord("q"):
        print("Received q. Exiting scene...")
        del C
        for obj in to_del:
            del obj
        return False

    else:
        print("Continuing...")
        return True
    
def detect_and_display(bot:ry.BotOp, camera_name, K, camera_rot, camera_coor, display=False):
    """
    Unified function for detection with optional display
    
    Args:
        display: If True, show visualization and return
    Returns:
        best_ball_world_coor: World coordinates of best ball or None
        best_circle: Pixel coordinates (x, y, radius) of best circle or None
    """
    # Capture and process
    rgb, depth, _ = bot.getImageDepthPcl(camera_name)
    mask_blur = det.filter_image_by_color(rgb)
    circles = det.circl_hough_transform(mask_blur)
    
    best_ball_world_coor = None
    best_circle = None
    
    # Process detection results
    if circles is not None and len(circles[0]) > 0:
        # Select best circle (largest radius)
        circles_array = circles[0]  # Shape: [n, 3]
        max_radius_idx = np.argmax(circles_array[:, 2])
        best_circle = np.uint16(np.around(circles_array[max_radius_idx]))
        
        # Get world coordinates
        best_ball_world_coor = det.get_world_from_camera(
            circle=best_circle, depth=depth, K=K,
            camera_rot=camera_rot, camera_coor=camera_coor
        )
        
        if best_ball_world_coor is not None:
            print(f"Best ball at world coordinates: {best_ball_world_coor} (radius: {best_circle[2]}px)")
    else:
        if not display:  # Only print in non-display mode
            print("No circles detected")
    
    # Display if requested
    if display:
        # For display: create array with just the best circle (or all circles)
        display_circles = None
        if best_circle is not None:
            display_circles = np.array([[best_circle]])  # Shape [1, 1, 3]
        elif circles is not None:
            display_circles = circles
        
        rgb_display = det.draw_detected_circles(rgb, display_circles)
        
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        axs[0].imshow(rgb, cmap='gray')
        axs[0].set_title("Original")
        axs[0].axis('off')
        
        axs[1].imshow(mask_blur, cmap='gray')
        axs[1].set_title("Color Mask")
        axs[1].axis('off')
        
        axs[2].imshow(rgb_display)
        axs[2].set_title("Detected Circle(s)")
        axs[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        return best_ball_world_coor, best_circle
    
    # Return results for non-display mode
    return best_ball_world_coor, best_circle


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
    if not check_config(C):
        exit(0)

    camera_pos = {"top": "cameraTopMiddle", "wrist": "cameraWrist"}
    if args.input in camera_pos:
        bot = ry.BotOp(C=C, useRealRobot=args.real)
        fxycxy = bot.getCameraFxycxy(camera_pos[args.input]) 
        fx, fy, cx, cy = fxycxy
        K = np.array([[fx, 0,  cx],
                    [0,  fy, cy],
                    [0,  0,  1.0]]) 
        camera_coor = C.getFrame(camera_pos[args.input]).getPosition()
        camera_rot = C.getFrame(camera_pos[args.input]).getRotationMatrix()
        print("Camera position: ", camera_coor)
        print("Camera rotation: \n", camera_rot)

        if args.display:   
            # inside detection loop
            detect_and_display(
                bot=bot,
                camera_name=camera_pos[args.input],
                K=K,
                camera_rot=camera_rot,
                camera_coor=camera_coor,
                display=True
            )
            return  
        else:
            while True: # This should be the detection loop
                best_ball, best_circle = detect_and_display(
                    bot=bot,
                    camera_name=camera_pos[args.input],
                    det=det,
                    K=K,
                    camera_rot=camera_rot,
                    camera_coor=camera_coor,
                    display=False
                )
                
                if best_ball is not None:
                    # Use for motion planning
                    #plan_to_target(best_ball)
                    pass
                
                time.sleep(0.033)  # ~30 Hz
        


    elif args.input == "standalone":
        camera = RealSense()
        camera.connect()
        camera.start_grabbing()
        rgb, depth = camera.acquire_rgb_and_depth()
    else:
        rgb, depth = None, None

    

    # Setup everything for path planning and execution
    q_home = C.getJointState()
    n_phases = 1
    komo = planner.setup_komo(C, n_phases)
    path = planner.get_path_from_komo(komo)
    if path is None:
        return

    if not args.botop or args.sim_only:
        # path = np.load("motion/ball_shot.npy")
        sim = SimWrapper(C, display_delay=0.1)
        # sim.close_gripper()
        sim.run_trajectory(path, n_phases)
        for _ in range(50):
            sim.step([], ry.ControlMode.none)  # step sim to observe
        sim.reset()

    else:
        bot = ry.BotOp(C, useRealRobot=args.real)
        bot.wait(C, forKeyPressed=False, forTimeToEnd=True)
        bot.moveTo(q_home)
        # Close gripper
        bot.gripperMove(ry.ArgWord._left, 0.)
        bot.wait(C, forKeyPressed=True, forTimeToEnd=False)
        # Run reference path
        times = np.linspace(n_phases / path.shape[0], n_phases, path.shape[-2])
        bot.move(path, times)
        bot.wait(C, forKeyPressed=True, forTimeToEnd=False)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sim-only", action="store_true", default=False, help="Only evaluate the simulator")
    p.add_argument("--botop", action="store_true", default=False, help="Use botop in addition to the simulator for a comparison.")
    p.add_argument("--real", action="store_true", default=False, help="Use this arg if real robot is used")  # Use this arg to run on the real robot    
    p.add_argument("--input", type=str, default="wrist", help="Select camera: wrist, top or standalone")
    p.add_argument("--display", action="store_true", default=False, help="Show camera view")
    p.add_argument("--aruco", action="store_true", default=False, help="Detect AruCo markers in the RGB image")
    p.add_argument("--segment", action="store_true", default=False, help="Run instance segmentation to locate an object")

    args = p.parse_args()

    main(args)

    print("Program exited cleanly.")
