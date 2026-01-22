import time
import threading
import argparse
import numpy as np
import robotic as ry
import matplotlib.pyplot as plt
from typing import Union, List, Any
import cv2 as cv

# ===== Own packages =====
from camera.real_sense import RealSense
from camera.bot_camera import RobotCamera

from simulation.sim_wrapper import SimWrapper
from simulation.scene_builder import SceneBuilder

from object_detector.detector import Detector
from motion import komo_planner as planner

def main(args):
    print("[Args: ", args, " ]")

    # Global parameters that can influence the friction, grasp force, etc.
    ry.params_add(
        {
            "physx/angularDamping": 0.1,
            "physx/defaultFriction": 100.0,
            "physx/defaultRestitution": 0.7,
            "botsim/verbose": 0,                # Increase verbosity to 2 for more info
        }
    )  

    # Build robot scene
    C = SceneBuilder(ry.raiPath("scenarios/pandaSingle.g"), "./config/config_gp.g").build()
    if C is None:
        return
    bot = ry.BotOp(C=C, useRealRobot=args.real)
    ball_position = None
    thread_camera = None

    if args.input == "standalone":
        camera = RealSense()
        print("Using default camera parameters.")
        camera.connect()
        camera.start_grabbing()
        rgb, depth = camera.acquire_rgb_and_depth()

    else:
        camera = RobotCamera(C, bot, args.input, args.display)
        camera.open(run_loop=True)
        exit(0)
        # if camera is not None:
        #     detector = Detector(camera, display=True)
        #     detector.run()


    # Setup everything for path planning and execution
    q_home = C.getJointState()
    n_phases = 1
    ball_position = ball_position if ball_position is not None else [-0.6, 0.2, 2]
    komo = planner.setup_komo(C, n_phases, ball_position)
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
        bot.wait(C, forKeyPressed=False, forTimeToEnd=True)
        bot.moveTo(q_home)
        # Close gripper
        bot.gripperMove(ry.ArgWord._left, 0.)
        bot.wait(C, forKeyPressed=True, forTimeToEnd=False)
        # Run reference path
        times = np.linspace(n_phases / path.shape[0], n_phases, path.shape[-2])


        bot.move(path, times)
        bot.wait(C, forKeyPressed=True, forTimeToEnd=False)

    if thread_camera:
        thread_camera.join(timeout=0.5)

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
