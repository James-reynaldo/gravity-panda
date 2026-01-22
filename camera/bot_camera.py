from typing import List
import robotic as ry
import threading
import numpy as np

class RobotCamera():
    def __init__(self, ry_Config:ry.Config,  robot:ry.BotOp, input_camera:str) -> None:
        self.C = ry_Config
        self.bot = robot
        self.camera_pos = {"top": "cameraTopMiddle", "wrist": "cameraWrist"}
        self.camera_used = None

        try:
            self.camera_used = self.camera_pos[input_camera]  
        except:
            print("ERROR: Camera position does not match robot's cameras [top, wrist].")
            return None

        # Initialize camera parameter with default values
        self.K = np.array([[1.0, 0,  1.0],
                    [0,  1.0, 1.0],
                    [0,  0,  1.0]]) 
        self.coordinates = np.array([0.0, 0.0, 0.0])
        self.rotation = np.eye(3)
        
        # --- Thread Management ---
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Not good practice but improves developing experience maybe add to a start method later
        self.set_parameters()
        
    def set_parameters(self):
        if self.camera_used is not None:
            fx, fy, cx, cy = self.bot.getCameraFxycxy(self.camera_used) 
            self.K = np.array([[fx, 0,  cx],
                                [0,  fy, cy],
                                [0,  0,  1.0]])
            self.coordinates = self.C.getFrame(self.camera_used).getPosition()
            self.rotation = self.C.getFrame(self.camera_used).getRotationMatrix()
        
    def grab_once(self):
        rgb, depth, _ = self.bot.getImageDepthPcl(self.camera_used)
        return rgb, depth

        



