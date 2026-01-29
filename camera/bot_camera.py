import robotic as ry
import threading
import numpy as np
import cv2 as cv
import os
class RobotCamera():
    def __init__(self, ry_Config:ry.Config,  robot:ry.BotOp, input_camera:str) -> None:
        self.C = ry_Config
        self.bot = robot
        self.camera_pos = {"top": "cameraTopMiddle", "wrist": "cameraWrist"}
        self.camera_used = None
        self.display = False
        self.rotate = False

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
        
    def set_parameters(self):
        if self.camera_used is not None:
            fx, fy, cx, cy = self.bot.getCameraFxycxy(self.camera_used) 
            self.K = np.array([[fx, 0,  cx],
                                [0,  fy, cy],
                                [0,  0,  1.0]])
            self.coordinates = self.C.getFrame(self.camera_used).getPosition()
            self.rotation = self.C.getFrame(self.camera_used).getRotationMatrix()
            

    def get_color_intrinsics(self):
        W = 1280
        H = 720
        dist = np.zeros(5)
        return self.K, dist, W, H
    
    def grab_once(self):
        rgb, depth, _ = self.bot.getImageDepthPcl(self.camera_used)
        rgb = cv.cvtColor(rgb, cv.COLOR_BGR2RGB)
        if self.rotate:
            rgb = cv.rotate(rgb, cv.ROTATE_180)
        return rgb, depth

    def get_rgb_and_depth(self):
        while not self._stop_event.is_set():
            rgb,depth = self.grab_once()
            yield rgb, depth
                
    def run(self, display = None):
        """Show camera view. Also, this is a small example of how to get the camera frame."""
        if display is not None:
            self.display = display

        for (rgb, depth) in self.get_rgb_and_depth():
            if self.display:
                cv.imshow("RobotCamera Feed", rgb)
                if cv.waitKey(10) & 0xFF == 27:
                    print("Exiting program...")
                    return



    def open(self, display:bool = False, rotate:bool = False):
        self.set_parameters()
        self.display = display
        self.rotate = rotate
            

    def close(self):
        self._stop_event.set()
        cv.destroyAllWindows()


    def save_frame_to_jpg(self, frame, filename, output_dir="/tmp/frames"):
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, filename)

        success = cv.imwrite(file_path, frame, [cv.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            raise IOError(f"Failed to write image to {file_path}")

        return file_path




