import numpy as np
import cv2 as cv


class Detector():
    def __init__(self, camera, display) -> None:
        # --- Setup camera ---
        self.camera = camera
        self.K = camera.K
        self.camera_coor = camera.coordinates
        self.camera_rot = camera.rotation

        # --- View reuslt ---
        self.display = display

    def run(self):
        for (rgb, depth) in self.camera.get_rgb_and_depth():

            cv.imshow("RobotCamera Feed", rgb)
            if cv.waitKey(10) & 0xFF == 27:
                print("Exiting program...")
                return


