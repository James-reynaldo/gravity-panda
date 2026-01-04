import time

import numpy as np
import robotic as ry
import matplotlib.pyplot as plt
import cv2 as cv

##################
from typing import Union
BALL_DIAMETER = 0.063  # meters

def gen_scene_ball():
    cfg = ry.Config()
    cfg.addFile(ry.raiPath("scenarios/pandaSingle.g"))
    cfg.delFrame("panda_collCameraWrist")
    cfg.addFile("assets/cfg0.g")

    # Load STL file
    from stl import mesh
    your_mesh = mesh.Mesh.from_file("/home/aldo/Documents/Gravity_panda/gravity-panda/03/assets/Gravity_Panda_Project_Cup.stl")
    

    scale_factor = 0.001  # Scale from mm to m
    your_mesh.vectors *= scale_factor
    # Get vertices and triangles
    vertices = your_mesh.vectors.reshape(-1, 3)  # Shape: (n*3, 3)
    triangles = np.arange(len(vertices)).reshape(-1, 3)  # Shape: (n, 3)

    # Add cup as STL mesh object
    cup = cfg.addFrame("cup")
    # cup.setShape(ry.ST.marker, 0.02)
    cup.setMesh(vertices, triangles)
    cup.setParent(cfg.getFrame("l_gripper"))
    cup.setJoint(ry.JT.rigid)
    cfg.attach('l_gripper','cup')
    cup.setRelativePose([0, 0, -0.05, 0.5, 0.5, 0.5, 0.5])

    # **Add coordinate axes to visualize the cup's frame**
    axes = cfg.addFrame("cup_axes", "cup")  # Child of cup frame
    axes.setShape(ry.ST.marker, [0.1])  # Size of axes (0.1m long)
    axes.setRelativePose([0, 0, 0, 1, 0, 0, 0])  # Identity pose

    #print joint names
    joint_names = cfg.getJointNames()
    print("Robot Joint Names:")
    for name in joint_names:
        print(name)

    cfg.getFrame("world").setShape(ry.ST.marker, 0.01)
    # print list of frames
    frames = cfg.getFrames()
    print("Frames in the configuration:")
    for frame in frames:
        print(f"- {frame.name}")

    # Set the robot to the specified joint positions
    joint_positions = [-0.0163, 0.0973, 0.0171, -2.4725, -0.0156, 2.5616, -0.7701]

    cfg.setJointState(joint_positions)

    finger1 = cfg.getFrame('l_panda_finger_joint1')

    print(f"Finger 1 joint type: {finger1.getJointType()}")
    q = cfg.getJointState()
    print("Initial joint positions:", q)
    return cfg


def quat_angle_error(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = np.clip(np.abs(np.dot(q1, q2)), -1.0, 1.0)
    return 2.0 * np.arccos(dot)


class SimWrapper:
    def __init__(self, cfg: ry.Config, sim_tau: float = 5e-2, display_delay: float = 1e-1, verbose: int = 2):
        """
        Wrapper for the simulation class in robotic.

        Params:
            cfg: The ry configuration object.
            tau_sim: the inverse frequency of the simulation.
            display_delay: how much delay the visualization receives if activated. Useful for debugging.
            verbose: In addition to console logs, this turns on the visualization of the sim (weird overload, I know).
                Use verbose = 2 to visualize the rollouts.
        """
        self.cfg = cfg
        self.sim = ry.Simulation(cfg, engine=ry.SimulationEngine.physx, verbose=verbose)
        self.sim_tau = sim_tau
        self.display_delay = display_delay
        self.verbose = verbose
        self.q_home = self.sim.getState()
        self.reset()

    def reset(self):
        self.sim.resetTime(0.)
        self.sim.resetSplineRef()
        self.sim.setState(*self.q_home)

    def step(self, u: Union[np.ndarray, list], mode: ry.ControlMode) -> None:
        self.sim.step(u, self.sim_tau, mode)
        if self.verbose > 1 and self.display_delay > 0:
            time.sleep(self.display_delay)

    def set_spline_ref(self, path: Union[np.ndarray, list], times: Union[np.ndarray, list]) -> None:
        self.sim.setSplineRef(path, times)

    def close_gripper(self, gripper_name: str = "l_gripper", tol: float = 1e-6):
        self.sim.moveGripper(gripper_name, 0.)
        while self.sim.getGripperWidth(gripper_name) >= tol:
            self.step([], ry.ControlMode.none)

    def run_trajectory(self, path: Union[np.ndarray, list], n_phases: int):
        """Run a full trajectory.

        Params:
            path: The full sequence of control targets.
            n_phases: How many phases should be used,
                determines the coarseness of the simulation.
        """
        path = np.atleast_2d(path)

        # 1.: Simulated trajectory rollout
        # This code snippet should complete the rollout of a trajectory in simulation

        #### BEGIN SOLUTION
        path_len = len(path)
        phases = np.linspace(n_phases / path_len, n_phases, path_len)
        self.set_spline_ref(path, phases)

        step = self.step
        mode = ry.ControlMode.spline
        n_steps = int(n_phases / self.sim_tau)

        for _ in range(n_steps):
            step([], mode)

        #### END SOLUTION


def straight_push(cfg: ry.Config) -> ry.KOMO:
    """Shoot the ball object along a straight line to the target.

    Args:
        cfg: The config.
    """
    # Compute push parameters
    table = cfg.getFrame("table")
    table_pos = table.getPosition()
    table_height = table_pos[2]
    print(f"Table height: {table_height}")
    ball_position = cfg.getFrame("ball").getPosition()
    ball_shape = cfg.getFrame("ball").getSize()

    #### BEGIN SOLUTION
    # Get gripper frame
    gripper = "l_gripper"
    
    # Get current gripper position
    current_pos = cfg.getFrame(gripper).getPosition()
    cfg.getFrame(gripper).setShape(ry.ST.marker, 0.02)
    print(f"Current gripper position: {current_pos}")
    
    # Create target position with z=1.0
    target_pos = np.array([current_pos[0], current_pos[1], table_height + 0.2])
    print(f"Target gripper position: {target_pos}")

    cfg.addFrame("target_pos")
    cfg.getFrame("target_pos").setPosition(target_pos)
    # cfg.getFrame("target_pos").setShape(ry.ST.marker, 0.02)
    

    # Setup komo
    komo = ry.KOMO()
    komo.setConfig(cfg, True)
    komo.setTiming(phases=1, slicesPerPhase=12, durationPerPhase=1.0, kOrder=1)

    # Collision avoidance
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)

    komo.addObjective([], ry.FS.positionDiff, ['l_gripper', 'target_pos'], ry.OT.eq, [1e1])

    # Regularization: stay close to natural configuration
    q_home = cfg.getJointState()
    komo.addObjective([], ry.FS.jointState, [], ry.OT.sos, [1e-1], q_home)
    komo.addObjective([], ry.FS.scalarProductXY, ['l_gripper', 'world'], ry.OT.eq, [1e1], [1])
    komo.addObjective([], ry.FS.scalarProductYX, ['l_gripper', 'world'], ry.OT.eq, [1e1], [-1])

    return komo


def main(args):
    cfg = gen_scene_ball()
    q_home = cfg.getJointState()
    cfg.view(pause=True)
    n_phases = 2

    if args.sim_only:
        path = np.load("assets/ball_shot.npy")

        sim = SimWrapper(cfg, display_delay=0.1)
        sim.close_gripper()
        sim.run_trajectory(path, n_phases=n_phases)
        for _ in range(50):
            sim.step([], ry.ControlMode.none)  # step sim to observe
    else:
        # Get komo & solve
        komo = straight_push(cfg)
        sol = ry.NLP_Solver()
        
        sol.setProblem(komo.nlp())
        sol.setOptions(damping=1e-1, verbose=0, stopTolerance=1e-2, stopInners=100, stopEvals=1000)
        ret = sol.solve(True)


        # komo.report(True)
        # komo.view(True)
        print(f"Solution found is {'' if ret.feasible else 'not'} feasible.")

        # Check komo in player
        # final_state = komo.getPathFrames()[-1]
        path = komo.getPath()
        # gripper_positions = []

        # for i in range(path.shape[0]):
        #     cfg.setJointState(path[i])
        #     gripper_pos = cfg.getFrame("l_gripper").getPosition()
        #     gripper_positions.append(gripper_pos)

        # gripper_positions = np.array(gripper_positions)

        # # Print the gripper positions
        # print("Gripper positions along the path:")
        # print("Time  X        Y        Z")
        # print("-" * 30)
        # for i, pos in enumerate(gripper_positions):
        #     print(f"{i:2d}: [{pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}]")

        # komo.view_play(False, delay=.1)

        # Check komo in sim
        if not args.botop:
            sim = SimWrapper(cfg, display_delay=0.1)
            sim.close_gripper()
            # sim.run_trajectory(path, n_phases=n_phases)
            for _ in range(100):
                sim.step([], ry.ControlMode.none)
            sim.reset()

        # Execution on botop
        else:
            bot = ry.BotOp(C=cfg, useRealRobot=args.real)
            # bot.wait(cfg, forKeyPressed=False, forTimeToEnd=True)
            # bot.moveTo(q_home)
            # # Close gripper
            # bot.gripperMove(ry.ArgWord._left, 0.)
            # bot.wait(cfg, forKeyPressed=True, forTimeToEnd=False)
            # # Run reference path
            # times = np.linspace(n_phases / path.shape[0], n_phases, path.shape[-2])
            # bot.move(path, times)
            # bot.wait(cfg, forKeyPressed=True, forTimeToEnd=False)

            # #get robot joint positions
            # joint_positions = bot.get_q()
            # print("Final joint positions on BotOp:")
            # for i, joint_pos in enumerate(joint_positions):
            #     print(f"Joint {i + 1}: {joint_pos:.4f} radians")

            cameras = {"top": "cameraTopMiddle", "wrist": "cameraWrist"}
            import threading
            import time
            
            stop_capture = False
            ball_positions = []  # Store detected ball positions


            def capture_loop():
                # plt.ion()  # Interactive mode on
                # fig, ax = plt.subplots(1, 1, figsize=(10,5))
                
                while not stop_capture:
                    try:
                        rgb, depth, _ = bot.getImageDepthPcl(cameras["top"])
                        hsv = cv.cvtColor(rgb, cv.COLOR_RGB2HSV)

                        # Adjust these values based on your specific blue
                        lower_blue = np.array([90, 50, 50])    # Dark blue
                        upper_blue = np.array([130, 255, 255]) # Light blue
                        
                        # Create mask for blue regions
                        blue_mask = cv.inRange(hsv, lower_blue, upper_blue)
                        print("Blue pixels:", np.count_nonzero(blue_mask))
                        # Bitwise-AND mask and original imageE
                        rgb_display = rgb.copy()
                        mask_blur = cv.medianBlur(blue_mask, 5)
                        circles = cv.HoughCircles(mask_blur, cv.HOUGH_GRADIENT, dp=1, minDist=100,
                                                param1=100, param2=15, minRadius=5, maxRadius=30)
                        if circles is not None:
                            circles = np.uint16(np.around(circles))
                            for i in circles[0, :]:
                                # draw the outer circle
                                cv.circle(rgb_display, (i[0], i[1]), i[2], (0, 255, 0), 2)
                                # draw the center of the circle
                                cv.circle(rgb_display, (i[0], i[1]), 2, (0, 0, 255), 3)
                                # print(f"Detected ball at pixel: u={i[0]}, v={i[1]}, radius={i[2]}")
                        # Display the image (optional, can comment out if not needed)
                        # cv.imshow('Ball Detection', rgb_display)
                        # cv.waitKey(1)  # Brief pause to update display
                        # Convert pixel to camera coordinates
                        fxycxy = bot.getCameraFxycxy(cameras["top"]) 
                            
                        fx, fy, cx, cy = fxycxy
                        # print("Camera Intrinsics: fx, fy, cx, cy =", fx, fy, cx, cy)
                        K = np.array([[fx, 0,  cx],
                                        [0,  fy, cy],
                                        [0,  0,  1.0]])
                        
                        R = np.array([
                            [-1,  0,  0],
                            [ 0,  0, -1],
                            [ 0, -1,  0]
                        ])
                        t = np.array([0, 3, 1.2]).reshape(3,1)  #camera position in world coordinates

                        if circles is not None:
                            for i in circles[0, :]:
                                u, v = i[0], i[1]
                                cam_depth = (BALL_DIAMETER*fx)/(2*i[2])  # Z = (D*fx)/(2*r)
                                x = (u - cx) * cam_depth / fx
                                y = (v - cy) * cam_depth / fy
                                # Store position
                                ball_positions.append({
                                    'time': time.time(),
                                    'cam_coords': (x, y, cam_depth),
                                    'pixel_coords': (u, v)
                                })
                                ball_cam_coords = np.array([x, y, cam_depth, 1]).reshape(4,1)
                                ball_world_coords = R.T @ ball_cam_coords[:3] + t
                                x_w, y_w, z_w = ball_world_coords.flatten()
                                print(f"Ball Camera CoordinatSimWraes: X:{x:.2f}m Y:{y:.2f}m Z:{cam_depth:.2f}m")
                                print(f"Ball World Coordinates: X:{x_w:.2f}m Y:{y_w:.2f}m Z:{z_w:.2f}m")
                        

                        time.sleep(0.05)  # Control capture rate
                    except Exception as e:
                        print(f"Error in capture thread: {e}")
                        break
            
            # Start capture thread
            print("Starting ball detection thread...")
            capture_thread = threading.Thread(target=capture_loop)
            capture_thread.start()


if __name__ == "__main__":
    from argparse import ArgumentParser

    p = ArgumentParser()
    p.add_argument("--sim-only", action="store_true", default=False, help="Only evaluate the simulator")
    p.add_argument("--botop", action="store_true", default=False, help="Use botop in addition to the simulator for a comparison.")
    p.add_argument("--real", action="store_true", default=False,
                   help="Use this arg if real robot is used")
    args = p.parse_args()

    print(args)

    if args.real and not args.botop:
        p.error("--real can only be used with --botop")

    main(args)
