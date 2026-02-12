import argparse
import robotic as ry
import numpy as np
from typing import List
import time

def setup_komo(C: ry.Config, n_phases, ball_position):
# Compute push parameters
    table = C.getFrame("table")
    table_height = table.getSize()[2] * 0.5
    cup = C.getFrame("cup")
    tol_fingertip = 0.1
    cup_size = cup.getSize()[0] + tol_fingertip

    if ball_position[2] < 0.7:
        target_height = table_height + 0.7                                            # 0.8 is more safe
    else: 
        target_height = table_height + ball_position[2]
    target_position = np.array([ball_position[0] + cup_size / 2, ball_position[1] - 0.1, target_height])

    # Chose end_effector
    end_effector = "l_gripper"
    # end_effector = "cup"

    C.getFrame('target').setPosition(target_position)

    # Setup komo
    komo = ry.KOMO()
    komo.setConfig(C, True)
    komo.setTiming(phases=n_phases, slicesPerPhase=8, durationPerPhase=1.0,kOrder=2)
  
    # Constrains on system
    komo.addControlObjective([], 2, 1e-1)
    komo.addControlObjective([], 1, 1e0)                                                # Penalize velocity
    komo.addControlObjective([], 0, 1e-1)                                               # Penalize joint motion magnitude
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, [1e1])                     # Do not exceed robots joint limits
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)                    # Do not self collide

    # Constraints on end_effector orientation
    # Camera facing upwards
    komo.addObjective([1], ry.FS.scalarProductZZ, ['world', end_effector], ry.OT.eq, [1e1], [0])
    komo.addObjective([1], ry.FS.scalarProductYX, ['world', end_effector], ry.OT.eq, [1e1], [-1])
    komo.addObjective([1], ry.FS.scalarProductXZ, ['world', end_effector], ry.OT.eq, [1e1], [1])

    # # Camera facing downwards
    # komo.addObjective([1], ry.FS.scalarProductZZ, ['world', end_effector], ry.OT.eq, [1e1], [0])
    # komo.addObjective([1], ry.FS.scalarProductYZ, ['world', end_effector], ry.OT.eq, [1e1], [0])
    # komo.addObjective([1], ry.FS.scalarProductXZ, [end_effector, 'world'], ry.OT.eq, [1e1], [0])


    # print("POSITION:::", C.getFrame('target').getPosition())    
    komo.addObjective([1], ry.FS.positionDiff, [end_effector,'target'], ry.OT.eq, [1e1])     # try to reach target immediately

    return komo

def get_path_from_komo(komo=None, display:bool=False):
    sol = ry.NLP_Solver()
    sol.setProblem(komo.nlp())
    sol.setOptions(damping=1e-1, verbose=0, stopTolerance=1e-2, stopInners=100, stopEvals=1000)
    ret = sol.solve(True)

    path = komo.getPath()

    # Show planned path
    if display:
        komo.view_play(False, delay=.1)
    
    # Check if solution is feasible
    if not ret.feasible:
        print("Solution is not feasible. Skipping...")
        return None
    print("Solution is feasible.")

    return path