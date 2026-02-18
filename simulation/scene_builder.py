import numpy as np
import robotic as ry
from typing import List, Any, Optional

class SceneBuilder():
    def __init__(self, scenario_path:str, scene_config_path:str ) -> None:
        """
        Build a ry-Scene from config file.
        
        """
        self.C :ry.Config = None
        self.scenario = scenario_path
        self.scene_config = scene_config_path

    def get_cup_quaternion(self, finger_rotation, cup_rotation):
        w1, x1, y1, z1 = finger_rotation
        w2, x2, y2, z2 = cup_rotation

        q_final = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

        return q_final

    def preview(self, to_del:List[Any] = [], message:str="Press q to quit (any other key to continue)") -> bool:
        user_key = self.C.view(pause=True, message=message)
        if user_key == ord("q"):
            print("Received q. Exiting scene...")
            del self.C
            for obj in to_del:
                del obj
            return False

        else:
            print("Continuing...")
            return True
        
    def build(self) -> Optional[ry.Config]:
        # Create ry Config
        self.C = ry.Config()

        # Build robot environment
        self.C.addFile(self.scenario)
        self.C.delFrame("panda_collCameraWrist")
        self.C.addFrame('world_marker').setShape(ry.ST.marker, [.1]).setPosition(self.C.getFrame('world').getPosition())
        
        # Add objects and camera from config
        self.C.addFile(self.scene_config)

        # Attach cup to robot
        self.C.addFrame("cup", "l_finger1")
        finger = self.C.getFrame("l_finger1")
        q_finger = finger.getQuaternion()
        q_90_rx = np.array([0.7071, 0.0, -0.7071, 0.0])
        q_cup = self.get_cup_quaternion(q_finger, q_90_rx)

        # Build cup as cylinder
        cup = self.C.getFrame("cup")
        cup.setShape(ry.ST.cylinder, [0.06, 0.06])
        cup.setColor([0.8, 0.3, 0.3])

        # Setup cup pose
        cup.setRelativePosition([-0.02, -0.035, 0.065]) 
        cup.setQuaternion(q_cup.tolist())
        self.C.addFrame('cup_marker').setShape(ry.ST.marker, [.1]).setPosition(cup.getPosition())

        if self.preview():
            return self.C
        
        return None

    