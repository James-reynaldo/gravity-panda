
import argparse
import numpy as np
import robotic as ry
import matplotlib.pyplot as plt
from typing import Union, List, Any

# ===== Own packages =====
from simulation.sim_wrapper import SimWrapper

def gen_scene():
    """Generate robot scene (environment) and return robotic-Config."""
    C = ry.Config()
    C.addFile(ry.raiPath("scenarios/pandaSingle.g"))
    C.delFrame("panda_collCameraWrist")
    C.addFile("../config/config_scene_3.g")
    
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
    
def quat_angle_error(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = np.clip(np.abs(np.dot(q1, q2)), -1.0, 1.0)
    return 2.0 * np.arccos(dot)

def straight_push(C: ry.Config) -> ry.KOMO:
    """Shoot the ball object along a straight line to the target.

    Args:
        C: The config.
    """
    # Compute push parameters
    table = C.getFrame("table")
    table_pos = table.getPosition()
    table_height = table.getSize()[2] * .5
    ball_position = C.getFrame("ball").getPosition()
    goal_post1 = C.getFrame("post1").getPosition()
    target_position = np.array([
        goal_post1[0],
        (C.getFrame("post2").getPosition()[1] + goal_post1[1]) / 2,
        ball_position[2]
    ])
    ball_shape = C.getFrame("ball").getSize()

    # Get current Gripper position
    gripper = "l_gripper"
    gripper_frame = C.getFrame(gripper)
    gripper_start_position = C.getJointState().copy()
   
    # Define push vector (direction) 
    dir_push_2d = target_position[:2] - ball_position[:2]
    dir_push = np.append(dir_push_2d, 0.0)
    length_push = np.linalg.norm(dir_push_2d)
    unit_push_2d = dir_push_2d / length_push
    unit_push = np.append(unit_push_2d, 0.0)

    push_direction_matrix = 1e2 * (np.eye(3) - np.outer(unit_push, unit_push))
 
    new_start = ball_position - unit_push * ball_shape * 5
    new_goal = target_position - unit_push * ball_shape * 8

    # Create Marker to show start and goal positions
    C.addFrame('unit_push'). setShape(ry.ST.marker, [.1]) .setPosition(unit_push)
    C.addFrame('start'). setShape(ry.ST.marker, [.1]) .setPosition(new_start)
    C.addFrame('goal'). setShape(ry.ST.marker, [.1]) .setPosition(new_goal)

    # Disable necessary contacts in optimization
    C.getFrame("ball").setContact(0)
    C.getFrame("post1").setContact(0)  
    C.getFrame("post2").setContact(0)

    # Setup komo
    komo = ry.KOMO()
    komo.setConfig(C, True)
    komo.setTiming(phases=2, slicesPerPhase=12, durationPerPhase=1,kOrder=1)
  
    # Add constraints
    komo.addControlObjective([], 1, 1e0) # Penalize velocity
    komo.addControlObjective([], 0, 1e-1) # Penalize jpint motion magnitude
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, [1e1])    # Do not exceed robots joint limits
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)
    komo.addObjective([1, 2], ry.FS.scalarProductYZ, ['start', gripper], ry.OT.eq, [1e1], [0])
    komo.addObjective([1, 2], ry.FS.scalarProductXZ, ['start', gripper], ry.OT.eq, [1e1], [0])
    komo.addObjective([1], ry.FS.scalarProductXX, ['unit_push', gripper], ry.OT.eq, [1e1], [0])

    # Only let gripper move in push direction
    komo.addObjective([1, 2], ry.FS.positionRel, [gripper, "start"], ry.OT.eq, push_direction_matrix, [0., 0., 0.])
    komo.addObjective([1, 2], ry.FS.vectorY, [gripper], ry.OT.eq, [1e1], unit_push)

    komo.addObjective([1], ry.FS.positionDiff, [gripper,'start'], ry.OT.eq, [1e1])  # Phase 1
    komo.addObjective([2], ry.FS.positionDiff, [gripper,'goal'], ry.OT.eq, [1e1])   # Phase 2

    return komo

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

    C = gen_scene()
    if not check_config(C):
        exit(0)

    q_home = C.getJointState()
    n_phases = 2

    if args.sim_only:
        path = np.load("assets/ball_shot.npy")

        sim = SimWrapper(C, display_delay=0.1)
        sim.close_gripper()
        sim.run_trajectory(path, n_phases=n_phases)
        for _ in range(50):
            sim.step([], ry.ControlMode.none)  # step sim to observe

    else:
        # Get komo & solve
        komo = straight_push(C)
        sol = ry.NLP_Solver()

        # # Verbose komo settings
        # komo.report(True)
        # komo.view(True)
        # print(komo.report())
        
        sol.setProblem(komo.nlp())
        sol.setOptions(damping=1e-1, verbose=0, stopTolerance=1e-2, stopInners=100, stopEvals=1000)
        ret = sol.solve(True)

        # Inform if solution is feasible
        print(f"Solution found is {'' if ret.feasible else 'not'} feasible.")

        path = komo.getPath()

        komo.view_play(False, delay=.1)

        # Check komo in sim
        if not args.botop:
            sim = SimWrapper(C, display_delay=0.1)
            sim.close_gripper()
            sim.run_trajectory(path, n_phases=n_phases)
            for _ in range(100):
                sim.step([], ry.ControlMode.none)
            sim.reset()

        # Execution on botop
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
    p.add_argument("--real", action="store_true", default=False, help="Use this arg if real robot is used")
    args = p.parse_args()

    if args.real and not args.botop:
        p.error("--real can only be used with --botop")

    main(args)
