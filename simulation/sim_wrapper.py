import time
import numpy as np
import robotic as ry
from typing import Union, List, Any

class SimWrapper:
    def __init__(self, C: ry.Config, sim_tau: float = 3e-2, display_delay: float = 1e-1, verbose: int = 2):
        """
        Wrapper for the simulation class in robotic.

        Params:
            C: The ry configuration object.
            tau_sim: the inverse frequency of the simulation.
            display_delay: how much delay the visualization receives if activated. Useful for debugging.
            verbose: In addition to console logs, this turns on the visualization of the sim (weird overload, I know).
                Use verbose = 2 to visualize the rollouts.
        """
        self.C = C
        self.sim = ry.Simulation(C, engine=ry.SimulationEngine.physx, verbose=verbose)
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
        path_len = len(path)
        phases = np.linspace(n_phases / path_len, n_phases, path_len)
        n_steps = int(n_phases / self.sim_tau)

        self.set_spline_ref(path, phases)

        for _ in range(n_steps):
            self.step([], ry.ControlMode.spline)
        


