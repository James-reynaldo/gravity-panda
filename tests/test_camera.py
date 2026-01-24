import argparse
import robotic as ry
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from camera.bot_camera import RobotCamera
from camera.real_sense import RealSense
from simulation.scene_builder import SceneBuilder

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
        print("Error failed to build scene. Exiting Program.")
        exit(0)
    bot = ry.BotOp(C=C, useRealRobot=False)

    if args.input == "standalone":
        camera = RealSense()
        print("Using default camera parameters.")
        camera.connect()
        camera.start_grabbing()
        rgb, depth = camera.acquire_rgb_and_depth()

    else:
        camera = RobotCamera(C, bot, args.input)
        camera.open(display=True)
        camera.run()



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input", type=str, default="wrist", help="Select camera: wrist, top or standalone")
    args = p.parse_args()

    main(args)

    print("Program exited cleanly.")