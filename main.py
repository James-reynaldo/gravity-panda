import time
import argparse
import numpy as np
import robotic as ry

# ===== Own packages =====
from camera.real_sense import RealSense
from camera.bot_camera import RobotCamera

from simulation.sim_wrapper import SimWrapper
from simulation.scene_builder import SceneBuilder

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

    # ===== Build simulation environment =====

    C = SceneBuilder(ry.raiPath("scenarios/pandasTable.g"), "./config/config_gp.g").build()
    if C is None:
        return
    bot = ry.BotOp(C=C, useRealRobot=args.real)

    # ==== Choose input camera =====
    if args.input == "standalone":
        camera = RealSense()
        print("Using default camera parameters.")
        camera.connect()
        camera.start_grabbing()

    else:
        camera = RobotCamera(C, bot, args.input)
        camera.open(display=args.display)

    # ===== Setup ball =====
    test_ball_positions = [[-1.2, 0.2, 2], [-0.5, 0.4, 1.4], [0.4, 0.2, 1.2], [0.6, 0.2, 1], [-0.3, 0.0, 0.8]]

    if  args.botop:
        q_home = C.getJointState()
        bot.wait(C, forKeyPressed=False, forTimeToEnd=True)
        bot.moveTo(q_home)
        bot.gripperMove(ry.ArgWord._left, 0.)
        bot.wait(C, forKeyPressed=False, forTimeToEnd=True)

    # ===== MAIN LOOP =====

    for i in range(1): #range(len(ball_positions)):

        # ===== Compute Robot Path =====
        n_phases = 1
        ball_position = test_ball_positions[i]
        ball = C.getFrame("ball")
        ball.setPosition(ball_position)
        komo = planner.setup_komo(C, n_phases, ball_position)
        path = planner.get_path_from_komo(komo, args.display)
        if path is None:
            continue

        # Execute path with botop or in simulation
        if  args.botop:
            times = np.linspace(n_phases / path.shape[0], n_phases, path.shape[-2])
            bot.move(path, times)
            bot.wait(C, forKeyPressed=False, forTimeToEnd=True)

        else:
            sim = SimWrapper(C, display_delay=0.1)
            sim.run_trajectory(path, n_phases)
            time.sleep(1)
            # sim.reset()

    camera.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="top", help="Select camera: wrist, top or standalone")
    p.add_argument("--botop", action="store_true", default=False, help="Use botop in addition to the simulator for a comparison.")
    p.add_argument("--real", action="store_true", default=False, help="Use this arg if real robot is used")  # Use this arg to run on the real robot    
    p.add_argument("--display", action="store_true", default=False, help="Show camera view")
    p.add_argument("--segment", action="store_true", default=False, help="Run instance segmentation to locate an object")

    args = p.parse_args()

    main(args)

    print("Program exited cleanly.")
