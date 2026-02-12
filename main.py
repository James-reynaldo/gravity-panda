import time
import argparse
import robotic as ry
import threading
import signal
import time

# === Own packages ===
from camera.real_sense import RealSense
from camera.bot_camera import RobotCamera
from object_detector.yolo_detector import BallDetector
from simulation.sim_wrapper import SimWrapper
from simulation.scene_builder import SceneBuilder
from motion import komo_planner as planner

_stop_event = threading.Event()

def handle_sigint(signum, frame):
    print("\nSIGINT received (Ctrl+C). Cleaning up...")
    global _stop_event
    _stop_event.set()

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

    global _stop_event

    # Register signal handler
    signal.signal(signal.SIGINT, handle_sigint)

    # === Build simulation environment ===
    C = SceneBuilder(ry.raiPath("scenarios/pandaSingle.g"), "./config/config_gp.g").build()

    if C is None:
        return
    
    q_home = C.getJointState()

    bot = ry.BotOp(C=C, useRealRobot=args.real)

    # === Choose input camera ===
    if args.input == "standalone":
        camera = RealSense()
        print("Using default camera parameters.")
        camera.connect()
        camera.start_grabbing()

    else:
        camera = RobotCamera(C, bot, args.input)
        camera.open(display=args.display, rotate=args.rotate)

    # === Initialize detector ===
    detector = BallDetector(camera=camera, display=args.display)
    if args.detectonly:
        detector.run()
        exit(0)

    # === Start from home === 
    if  args.botop:
        print("Moving home.")
        bot.wait(C, forKeyPressed=False, forTimeToEnd=True)
        bot.moveTo(q_home)
        bot.gripperMove(ry.ArgWord._left, 0.079)
        bot.wait(C, forKeyPressed=False, forTimeToEnd=True)

    # Set parameters for testing and komo
    test_ball_positions = [[-0.139, 0.2, 1.5], [-0.5, 0.4, 1.4], [0.1, 0.2, 1.2], [-0.6, 0.2, 1], [-0.3, 0.4, 0.8]]
    n_phases = 1

    
    print("Start loop...")

    C.addFrame('target').setShape(ry.ST.marker, [.1]) 
    C.addFrame('world_frame').setShape(ry.ST.marker, [.1]).setPosition(C.getFrame('world').getPosition())

    # === Grab cup ===
    cup_postition = [-0.35, 0.64, .71]
    komo = planner.setup_komo(C, 1, cup_postition)
    path = planner.get_path_from_komo(komo, True)
    if path is None:
        exit(0)
    if  args.botop:
        times = [3]
        bot.move(path, times)
        bot.wait(C, forKeyPressed=False, forTimeToEnd=True)
        bot.gripperClose(ry.ArgWord._left)
        bot.wait(C, forKeyPressed=False, forTimeToEnd=True)

    # =================
    # === MAIN LOOP ===
    # =================
    while not _stop_event.is_set():
        # Grab frame
        rgb, depth = camera.grab_once()
        # Predict ball
        results = detector.predict_frame(frame=rgb, confidence=0.4)
        ball = detector.process_results(results)
        if ball is None:
            print("No ball detected.")
            continue
        
        # Get real world coordinates from pixels
        ball_position = detector.get_world_from_camera(ball, depth, camera.K, camera.rotation, camera.coordinates)

    # for ball_position in test_ball_positions:
        if ball_position is None:
            print("No ball position.")
            time.sleep(0.1)
            continue

        # Compute robot path with komo
        ball_position = [ball_position[0], ball_position[1], 0.9, 0, 0, 0]
        komo = planner.setup_komo(C, 1, ball_position)
        path = planner.get_path_from_komo(komo, True)
        
        if path is None:
            # Solution not feasible -> skip.
            continue

        # Execute path with botop or in simulation
        if  args.botop:
            times = [2] # 2 sec until target is reached
            bot.move(path, times)
            bot.wait(C, forKeyPressed=False, forTimeToEnd=True)

        else:
            sim = SimWrapper(C, display_delay=0.1)
            sim.run_trajectory(path, n_phases)
            time.sleep(1)
            # sim.reset()
        time.sleep(1)
        komo.clearObjectives()

    

    # === Put cup back ===
    cup_postition = [-0.35, 0.64, .71]
    komo = planner.setup_komo(C, 1, cup_postition)
    path = planner.get_path_from_komo(komo, True)
    if path is None:
        exit(0)
    if  args.botop:
        print("Placing cup...")
        times = [3]
        bot.move(path, times)
        bot.wait(C, forKeyPressed=False, forTimeToEnd=True)
        bot.gripperMove(ry.ArgWord._left, 0.07)

        # ===  End at home === 
        print("Returning home...")
        bot.wait(C, forKeyPressed=False, forTimeToEnd=True)
        bot.moveTo(q_home)
        bot.wait(C, forKeyPressed=False, forTimeToEnd=True)

    camera.close()

    del bot
    del C



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="top", help="Select camera: wrist, top or standalone")
    p.add_argument("--botop", action="store_true", default=False, help="Use botop in addition to the simulator for a comparison.")
    p.add_argument("--real", action="store_true", default=False, help="Use this arg if real robot is used")  # Use this arg to run on the real robot    
    p.add_argument("--display", action="store_true", default=False, help="Show camera view")
    p.add_argument("--segment", action="store_true", default=False, help="Run instance segmentation to locate an object")
    p.add_argument("--detectonly", action="store_true", default=False, help="")
    p.add_argument("--rotate", action="store_true", default=False, help="")

    args = p.parse_args()

    main(args)

    print("Program exited cleanly.")
