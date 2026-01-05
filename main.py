import time

import numpy as np
import robotic as ry
from typing import Union


def gen_scene_ball():
    cfg = ry.Config()
    cfg.addFile(ry.raiPath("scenarios/pandaSingle.g"))
    cfg.delFrame("panda_collCameraWrist")
    cfg.addFile("assets/cfg0.g")
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
        self.sim.setState(self.q_home)

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

        # Simulated trajectory rollout
        path_len = len(path)
        phases = np.linspace(n_phases / path_len, n_phases, path_len)
        self.set_spline_ref(path, phases)

        step = self.step
        mode = ry.ControlMode.spline
        n_steps = int(n_phases / self.sim_tau)

        for _ in range(n_steps):
            step([], mode)



def build_motion(cfg: ry.Config) -> ry.KOMO:
    """Shoot the ball object along a straight line to the target.

    Args:
        cfg: The config.
    """
    # Compute parameters
    table = cfg.getFrame("table")
    table_pos = table.getPosition()
    table_height = table.getSize()[2] * .5
    ball_position = cfg.getFrame("ball").getPosition()
    ball_shape = cfg.getFrame("ball").getSize()

    # Get current Gripper position
    gripper = "l_gripper"
    gripper_frame = cfg.getFrame(gripper)
    gripper_start_position = cfg.getJointState().copy()
   
    print(ball_shape)

    # Compute where ball impacts to table
    ball_impact = [ball_position[0], ball_position[1], 2*table_height]
    vec_fall = ball_impact - ball_position
    travel_fall = np.linalg.norm(vec_fall)
    unit_fall = vec_fall / travel_fall
    fall_direction_matrix = 1e2 * (np.eye(3) - np.outer(unit_fall, unit_fall))

    # Create Marker to show start and goal positions
    cfg.addFrame('unit_fall'). setShape(ry.ST.marker, [.1]) .setPosition(unit_fall)
    cfg.addFrame('ball_impact'). setShape(ry.ST.marker, [.1]) .setPosition(ball_impact)

    # Disable necessary contacts in optimization
    cfg.getFrame("ball").setContact(0)

    return build_komo(cfg, gripper = gripper, unit_fall = unit_fall, fall_dir_mat = fall_direction_matrix)
    

def build_komo(cfg, **kwargs):
    gripper = kwargs.get('gripper')
    unit_fall = kwargs.get('unit_fall')
    fall_dir_mat = kwargs.get('fall_dir_mat')
    # Setup komo
    komo = ry.KOMO()
    komo.setConfig(cfg, True)
    komo.setTiming(phases=2, slicesPerPhase=12, durationPerPhase=1,kOrder=1)
  
    # Add constraints
    komo.addControlObjective([], 1, 1e0) # Penalize velocity
    komo.addControlObjective([], 0, 1e-1) # Penalize jpint motion magnitude

    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, [1e1])    # Do not exceed robots joint limits
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)

    # komo.addObjective([1,2], ry.FS.vectorXDiff, [gripper, 'goal'], ry.OT.eq, [1e1])
    # komo.addObjective([1], ry.FS.scalarProductYZ, [gripper, 'unit_fall'], ry.OT.eq, [1e1], [0])
    # komo.addObjective([1], ry.FS.scalarProductXZ, [gripper, 'unit_fall'], ry.OT.eq, [1e1], [0])
    # komo.addObjective([1], ry.FS.scalarProductXX, ['unit_push', gripper], ry.OT.eq, [1e1], [0])

    # Only let gripper move in push direction
    komo.addObjective([2], ry.FS.positionRel, [gripper, "unit_fall"], ry.OT.eq, fall_dir_mat, [0., 0., 0.])
    komo.addObjective([2], ry.FS.vectorZ, [gripper], ry.OT.eq, [1e1], unit_fall)

    # Phase 2: End contact (push complete)
    komo.addObjective([1], ry.FS.positionDiff, [gripper,'ball_impact'], ry.OT.eq, [1e1])
    komo.addObjective([2], ry.FS.positionDiff, [gripper,'ball'], ry.OT.eq, [1e1])

    #### END SOLUTION
    # print(komo.report())

    return komo


def main(args):
    cfg = gen_scene_ball()
    q_home = cfg.getJointState()
    cfg.view(pause=True)
    n_phases = 2

    if args.sim_only:
        path = np.load("assets/ball_shot.npy")

        sim = SimWrapper(cfg, display_delay=0.1)
        # sim.close_gripper()
        sim.run_trajectory(path, n_phases=n_phases)
        for _ in range(50):
            sim.step([], ry.ControlMode.none)  # step sim to observe
    else:
        # Get komo & solve
         
        komo = build_motion(cfg)
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

        komo.view_play(False, delay=.1)

        # Check komo in sim
        if not args.botop:
            sim = SimWrapper(cfg, display_delay=0.1)
            # sim.close_gripper()
            sim.run_trajectory(path, n_phases=n_phases)
            for _ in range(100):
                sim.step([], ry.ControlMode.none)
            sim.reset()

        # Execution on botop
        else:
            bot = ry.BotOp(C=cfg, useRealRobot=args.real)
            bot.wait(cfg, forKeyPressed=False, forTimeToEnd=True)
            bot.moveTo(q_home)
            # Close gripper
            bot.gripperMove(ry.ArgWord._left, 0.)
            bot.wait(cfg, forKeyPressed=True, forTimeToEnd=False)
            # Run reference path
            times = np.linspace(n_phases / path.shape[0], n_phases, path.shape[-2])
            bot.move(path, times)
            bot.wait(cfg, forKeyPressed=False, forTimeToEnd=True)


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
