import cv2 as cv
import numpy as np
from typing import Tuple, Optional
from camera.real_sense import RealSense
from camera.bot_camera import RobotCamera
from ultralytics import YOLO

class Detector():
    def __init__(self, camera: RealSense|RobotCamera, kernel_size=(5,5), display:bool=False) -> None:
        # --- Image Color Filter ---
        self.color_lower = np.array([90, 50, 50])
        self.color_upper = np.array([130, 255, 170])
        self.blur_parameter = 5
        self.kernel_size = kernel_size
        self.min_area = 100

        # --- Camera parameters ---
        self.K = camera.K
        self.camera_rot = camera.rotation
        self.camera_coord = camera.coordinates

    # ===== Frame Processing =====

    def filter_image(self, rgb) -> np.ndarray:
        """ Filter the image to isolate the ball based on color. """

        # Color Filtering + Blurring
        hsv = cv.cvtColor(rgb, cv.COLOR_RGB2HSV)
        mask = cv.inRange(hsv, self.color_lower, self.color_upper)
        mask_blur = cv.medianBlur(mask, self.blur_parameter)

        # Seperate touching objects
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, self.kernel_size)
        mask_sep = cv.morphologyEx(mask_blur, cv.MORPH_OPEN, kernel)
        return mask_sep
    
    def find_circle_by_contours(self, mask: np.ndarray) -> list:
        """ Find circles in a mask using contour analysis. """

        cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv.contourArea(c) > self.min_area]               # Keep only large blobs
        print("Found {} contours after area filtering.".format(len(cnts)))
        candidates = []

        for c in cnts:
            area = cv.contourArea(c)
            perimeter = cv.arcLength(c, True)
            if perimeter == 0:
                continue
            # Threshold for circlarity of object shape
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            if circularity > 0.4:
                candidates.append(c)
        return candidates

    def get_world_from_camera(
        self,
        circle: Tuple[int, int, int],
        depth: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """
        Convert circle's camera coordinates to world coordinates.
        
        Args:
            circle: (center_x, center_y, radius)
            depth: Depth image (same resolution as RGB), can be None
            K: Camera intrinsic matrix (3x3)
            camera_rot: Camera rotation matrix (3x3)
            camera_coor: Camera position in world coordinates (3,) or (3,1)
        
        Returns:
            World coordinates (3,) or None if invalid
        """
        # Validate inputs
        if len(circle) != 3 or self.K.shape != (3, 3) or self.camera_rot.shape != (3, 3):
            print("Error: Invalid input shapes")
            return None
        
        pixel_u, pixel_v = int(circle[0]), int(circle[1])
        
        # Get depth with bounds checking
        if depth is not None:
            if not (0 <= pixel_v < depth.shape[0] and 0 <= pixel_u < depth.shape[1]):
                print(f"Error: Pixel ({pixel_u}, {pixel_v}) out of bounds")
                return None
            
            depth_value = float(depth[pixel_v, pixel_u])
            if depth_value <= 0 or not np.isfinite(depth_value):
                print(f"Warning: Invalid depth {depth_value} at ({pixel_u}, {pixel_v})")
                return None
        else:
            depth_value = 1.0
        
        # Prevent division by zero
        if self.K[0, 0] == 0 or self.K[1, 1] == 0:
            print("Error: Zero focal length")
            return None
        
        # Convert to camera coordinates (vectorized)
        camera_coordinates = np.array([
            (pixel_u - self.K[0, 2]) * depth_value / self.K[0, 0],
            (pixel_v - self.K[1, 2]) * depth_value / self.K[1, 1],
            depth_value
        ], dtype=np.float64)
        
        # Transform to world coordinates

        print(f"Detected pixel: ({pixel_u}, {pixel_v})")
        print(f"Depth value: {depth_value}")
        print(f"Camera coordinates: {camera_coordinates}")
        
        print(self.camera_rot.T @ [-0.6, -2.8, 0.8])
        
        return self.camera_rot.T @ camera_coordinates + self.camera_coord.flatten()

    def detect(
        self,
        rgb: np.ndarray,
        depth: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int]], Optional[np.ndarray]]:
        """
        Detect ball in frame and return world coordinates.
        
        Returns:
            (world_coordinates, circle, filtered_mask) - all None if detection fails
        """
        # Validate and filter image
        if rgb is None or rgb.size == 0:
            return None, None, None
        print(f"Image shape: {rgb.shape}")
        try:
            filtered_mask = self.filter_image(rgb)
            if filtered_mask is None:
                return None, None, None
            
            candidates = self.find_circle_by_contours(filtered_mask)
            if not candidates:
                print("No ball candidates found")
                return None, None, filtered_mask
            
            # Get best candidate and compute circle
            best_contour = max(candidates, key=cv.contourArea)
            (circle_x, circle_y), circle_radius = cv.minEnclosingCircle(best_contour)
            
            # Validate circle
            if circle_radius <= 0 or not (np.isfinite(circle_x) and np.isfinite(circle_y)):
                print(f"Invalid circle: center=({circle_x}, {circle_y}), radius={circle_radius}")
                return None, None, filtered_mask
            
            detected_circle = (int(circle_x), int(circle_y), int(circle_radius))
            
            # Convert to world coordinates
            world_coordinates = self.get_world_from_camera(detected_circle, depth)
            print(f"World coordinates: {world_coordinates}")
            return world_coordinates, detected_circle, filtered_mask
            
        except Exception as error:
            print(f"Detection error: {error}")
            return None, None, None

    # ===== YOLO =====        

    def predict_frame(self, model:YOLO, frame, confidence):
        """Predicts objects in a camera frame and returns the predicted results."""
        try:
            return model.predict(frame, verbose=False, conf=confidence)
        except ZeroDivisionError:
            print("Zero division error occurred during YOLO prediciton.")
            return  None 
        
    def process_results(self, results):
        """Process results form YOLO prediction."""

        comp_list = []

        for result in results:
            for box in result.boxes:
                component = {
                    "name": result.names[int(box.cls[0])],
                    "bbox": box.xyxy[0].tolist(),
                    "confidence": float(box.conf[0])
                }
                comp_list.append(component)

            
        return comp_list