import time
import threading
import argparse
import numpy as np
import robotic as ry
import matplotlib.pyplot as plt
from typing import Union, List, Any
import traceback

# ===== Own packages =====
from camera.real_sense import RealSense
from object_detector.detector import Camera
from simulation.sim_wrapper import SimWrapper
from motion import komo_planner as planner



def gen_scene():
    """Generate robot scene (environment) and return robotic-Config."""
    C = ry.Config()
    C.addFile(ry.raiPath("scenarios/pandasTable.g"))
    # C.delFrame("panda_collCameraWrist")
    C.addFile("config/config_gp.g")

    # Attach cup as basic cylinder to robot gripper
    C.addFrame("cup", "l_finger1")
    finger = C.getFrame("l_finger1")
    finger_orientation = finger.getQuaternion()
    cup = C.getFrame("cup")
    cup.setShape(ry.ST.cylinder, [0.06, 0.06])
    cup.setColor([0.8, 0.3, 0.3])
    
    rot_90_deg_rx = np.array([0.7071, 0.0, -0.7071, 0.0])

    w1, x1, y1, z1 = finger_orientation
    w2, x2, y2, z2 = rot_90_deg_rx

    q_final = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

    # Set cup pose
    cup.setRelativePosition([-0.02, -0.035, 0.065]) 
    cup.setQuaternion(q_final.tolist())

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
    ball_position = None
    last_ball_position = None
    # Setup everything for path planning and execution
    q_home = C.getJointState()
    n_phases = 1
    
    if args.input in camera_pos:
        bot = ry.BotOp(C=C, useRealRobot=args.real)
        # fxycxy = bot.getCameraFxycxy(camera_pos[args.input]) 
        # fx, fy, cx, cy = fxycxy
        # K = np.array([[fx, 0,  cx],
        #             [0,  fy, cy],
        #             [0,  0,  1.0]]) 
        # camera_coor = C.getFrame(camera_pos[args.input]).getPosition()
        # camera_rot = C.getFrame(camera_pos[args.input]).getRotationMatrix()
        # print("Camera position: ", camera_coor)
        # print("Camera rotation: \n", camera_rot)

        home_position = [-0.6, 0.2, 2]
        komo = planner.setup_komo(C, n_phases, home_position)
        path = planner.get_path_from_komo(komo)

        bot.wait(C, forKeyPressed=False, forTimeToEnd=True)
        bot.moveTo(q_home)
        # Close gripper
        bot.gripperMove(ry.ArgWord._left, 0.)
        bot.wait(C, forKeyPressed=True, forTimeToEnd=False)
        # Run reference path
        times = np.linspace(n_phases / path.shape[0], n_phases, path.shape[-2])


        bot.move(path, times)
        bot.wait(C, forKeyPressed=True, forTimeToEnd=False)

        # thread_camera = threading.Thread(target=do_camera, args=[args, bot, camera_pos, K, camera_rot, camera_coor, ball_position], daemon=True)
        # thread_camera.start()
        camera = Camera(
            C=C, 
            input=args.input, 
            bot=bot, 
            camera_pos= camera_pos, 
            display=True,
            display_target="rgb")
        camera.start()
        is_moving = False
        plan_cooldown = 0.1  # seconds
        last_plan_time = -np.inf
        
        try:
            if camera.display:
                # Matplotlib will handle updates via animation
                plt.show(block=False)  # This blocks until window is closed
                try:
                    while camera.running:
                        current_time = time.time()
                        ball_position = camera.ball_position 
                        
                        # Check motion status FIRST (outside ball check)
                        time_to_end = bot.getTimeToEnd()
                        if time_to_end > 0:
                            is_moving = True
                            print(f"Robot moving, TTE: {time_to_end:.2f}s")
                        else:
                            if is_moving:
                                print("Motion completed")
                            is_moving = False
                        if ball_position is not None:
                            # Check if ball moved enough to replan
                            should_replan = (
                                not is_moving and  # Robot not moving
                                (current_time - last_plan_time > plan_cooldown) and  # Cooldown passed
                                (last_ball_position is None or  # First detection
                                np.linalg.norm(np.array(ball_position) 
                                            - np.array(last_ball_position)) > 0.02)
                            )
                            if should_replan:
                                print(f"Ball moved to: {ball_position}")
                                # === CRITICAL FIX ===
                                # Sync C with actual robot state
                                bot.sync(C, waitTime=0.01)
                                # Now C matches where robot ACTUALLY is!
                                last_ball_position = ball_position.copy()
                                last_plan_time = time.time()
                                # Plan new path
                                komo = planner.setup_komo(C, n_phases, ball_position)
                                path = planner.get_path_from_komo(komo)
                                if path is not None:
                                    # Run reference path
                                    times = np.linspace(n_phases / path.shape[0], n_phases, path.shape[-2]) #Fixes duration to only 1s
                                    bot.move(path, times)
                                    
                                    # Update state
                                    is_moving = True
                                    TTE = bot.getTimeToEnd()
                                    print(f"Started motion, TTE: {TTE:.2f}s")
                        else:
                            print("NO BALL DETECTED...")

                        # Allow matplotlib to update
                        plt.pause(0.1)
                       
                except KeyboardInterrupt:
                    print("\nInterrupted by user")
            else:
                # If not displaying, just keep running
                while True:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            # Print the full traceback
            traceback.print_exc()  # Prints to stderr
            
            # Or get traceback as string
            error_msg = traceback.format_exc()
            print(f"ERROR DETAILS:\n{error_msg}")
        finally:
            camera.shutdown()
            print("Program exited cleanly.")


    elif args.input == "standalone":
        camera = RealSense()
        camera.connect()
        camera.start_grabbing()
        rgb, depth = camera.acquire_rgb_and_depth()
    else:
        rgb, depth = None, None
    

    # Setup everything for path planning and execution
    # q_home = C.getJointState()
    # n_phases = 1
    # ball_position = camera.ball_position if camera.ball_position is not None else [-0.6, 0.2, 2]
    # komo = planner.setup_komo(C, n_phases, ball_position)
    # path = planner.get_path_from_komo(komo)
    if path is None:
        return

    if not args.botop or args.sim_only:
        # path = np.load("motion/ball_shot.npy")
        sim = SimWrapper(C, display_delay=0.1)
        # sim.close_gripper()
        sim.run_trajectory(path, n_phases)
        for _ in range(50):
            sim.step([], ry.ControlMode.none)
        sim.reset()

    else:
        # bot = ry.BotOp(C, useRealRobot=args.real)
        # bot.wait(C, forKeyPressed=False, forTimeToEnd=True)
        # bot.moveTo(q_home)
        # # Close gripper
        # bot.gripperMove(ry.ArgWord._left, 0.)
        # bot.wait(C, forKeyPressed=True, forTimeToEnd=False)
        # # Run reference path
        # times = np.linspace(n_phases / path.shape[0], n_phases, path.shape[-2])


        # bot.move(path, times)
        # bot.wait(C, forKeyPressed=True, forTimeToEnd=False)
        pass

    # if thread_camera:
    #     thread_camera.join(timeout=0.5)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sim-only", action="store_true", default=False, help="Only evaluate the simulator")
    p.add_argument("--botop", action="store_true", default=False, help="Use botop in addition to the simulator for a comparison.")
    p.add_argument("--real", action="store_true", default=False, help="Use this arg if real robot is used")  
    p.add_argument("--input", type=str, default="wrist", help="Select camera: wrist, top or standalone")
    p.add_argument("--display", action="store_true", default=False, help="Show camera view")
    # p.add_argument("--segment", action="store_true", default=False, help="Run instance segmentation to locate an object")

    args = p.parse_args()

    main(args)

    print("Program exited cleanly.")
