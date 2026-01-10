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
    #  attach a cylinder to the robot left finger
    C.addFrame("cup", "l_finger1")
    cup = C.getFrame("cup")
    cup.setShape(ry.ST.cylinder, [0.04, 0.04])
    cup.setColor([0.8, 0.3, 0.3])
    cup.setRelativePosition([0.05, -0.02, 0])
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
    
def camera_stream(bot: ry.BotOp, camera_name, max_frames=None, single_frame=False):
    """
    Generator that yields camera frames with optional limit
    
    Args:
        bot: BotOp instance
        camera_name: Name of the camera to use
        max_frames: Optional maximum number of frames to yield
    Yields:
        Tuple of (frame_id, rgb, depth)
    """
    if single_frame:
        rgb, depth, _ = bot.getImageDepthPcl(camera_name)
        yield 0, rgb, depth
        return
    else:
        frame_id = 0
        while True:
            if max_frames is not None and frame_id >= max_frames:
                break
                
            rgb, depth, _ = bot.getImageDepthPcl(camera_name)
            yield frame_id, rgb, depth
            frame_id += 1


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

        if args.display: #display only one frame
            # inside detection loop
            stream = camera_stream(bot, camera_pos[args.input], single_frame=True)
            for result in det.detect_stream(stream, K, camera_rot, camera_coor, display=True):
                world_coor, _ = result
                if world_coor is not None:
                    pass
                else:
                    print("No ball detected in the frame.")
            return  
        else:
            stream = camera_stream(bot, camera_pos[args.input])
            for result in det.detect_stream(stream, K, camera_rot, camera_coor, display=False):
                world_coor, _ = result
                if world_coor is not None:
                    # Do something with the detection
                    pass
                else:
                    print("No ball detected in the frame.")

    elif args.input == "standalone":
        camera = RealSense()
        camera.connect()
        camera.start_grabbing()
        rgb, depth = camera.acquire_rgb_and_depth()
    else:
        rgb, depth = None, None

    bot.sync
    

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
