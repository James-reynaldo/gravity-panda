import numpy as np
import cv2 as cv
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class Component:
    name: str
    confidence: float
    bbox: List[float] = field(default_factory=list)
    
    @property
    def center(self) -> Tuple[float, float]:
        if len(self.bbox) != 4:
            raise ValueError("bbox must contain exactly four values: [x_min, y_min, x_max, y_max]")
        x_min, y_min, x_max, y_max = self.bbox
        return ((x_min + x_max) / 2, (y_min + y_max) / 2)



class BallDetector():
    def __init__(self, camera, modelpath="./yolo/best.pt", display=False) -> None:
        # --- Setup camera ---
        self.camera = camera
        self.K = camera.K
        self.camera_coor = camera.coordinates
        self.camera_rot = camera.rotation

        # --- View reuslt ---
        self.display = display
        self.model = YOLO(modelpath)


 
    

    def run(self):
        for (frame, depth_frame) in self.camera.get_rgb_and_depth_frame():

            # YOLO prediction
            results = self.predict_frame(frame, confidence=0.5)
            ball_list = self.process_results(results)
            if self.display:
                self.show_result(frame, ball_list)
                if cv.waitKey(1) & 0xFF == 27:
                    break
        cv.destroyAllWindows()
                
        # TODO: Detect ball in frame
            ## preprocess frame
            ## look for circles with opencv
            ## select circle

        # TODO: Get Ball center pixel coordinates

        # TODO: Calculate real world coordinates of ball

        # TODO: Return ball real world coordinates to robot



    def predict_frame(self, frame, confidence):
        """Predict objects in a camera frame and returns the predicted results."""
        try:
            results = self.model.predict(frame, verbose=False, conf=confidence)
            return results
        except ZeroDivisionError:
            print("Zero division error occurred during YOLO prediction.")
            return None
        except Exception as e:
            print(f"YOLO prediction error: {e}")
            return None

    def process_results(self, results) -> List[Component]:
        """Process results from YOLO prediction."""
        if results is None:
            return []

        comp_list = []

        for result in results:
            names = result.names  # FIX: cache names once
            for box in result.boxes:
                cls_id = int(box.cls[0])
                component = Component(names[cls_id], float(box.conf[0]), box.xyxy[0].tolist())
                if component.name in ["ball", "sports ball"]:
                    comp_list.append(component)
        return comp_list
    
    def show_result(self, frame, ball_list):
        # Draw detections
        for ball in ball_list:
            x1, y1, x2, y2 = map(int, ball.bbox)
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(frame, f"{ball.name} {ball.confidence:.2f}",
                        (x1, y1 - 10),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 0, 0), 2)
            # Draw center point
            cx, cy = map(int, ball.center)
            cv.circle(frame, (cx, cy), 2, (0, 255, 255), -1)


        cv.imshow("Blue Ball Detection", frame)

    def get_world_from_camera(
        self,
        ball:Component,
        depth_frame: np.ndarray,
    ) -> Optional[np.ndarray]:

        # Validate inputs
        if self.K.shape != (3, 3) or self.camera_rot.shape != (3, 3):
            print("Error: Invalid input shapes")
            return None
        
        pixel_u, pixel_v = map(int, ball.center)
        
        # Get depth_frame with bounds checking
        if depth_frame is  None:
            return None
        if not (0 <= pixel_v < depth_frame.shape[0] and 0 <= pixel_u < depth_frame.shape[1]):
            print(f"Error: Pixel values ({pixel_u}, {pixel_v}) out of depth frame limits.")
            return None
            
        depth_frame_value = float(depth_frame[pixel_v, pixel_u])
        
        if depth_frame_value <= 0 or not np.isfinite(depth_frame_value):
            print(f"Warning: Invalid depth_frame {depth_frame_value} at ({pixel_u}, {pixel_v})")
            return None
        
        # Prevent division by zero
        if self.K[0, 0] == 0 or self.K[1, 1] == 0:
            print("Error: Zero focal length")
            return None
        
        # Convert to camera coordinates (vectorized)
        camera_coordinates = np.array([
            (pixel_u - self.K[0, 2]) * depth_frame_value / self.K[0, 0],
            (pixel_v - self.K[1, 2]) * depth_frame_value / self.K[1, 1],
            depth_frame_value
        ], dtype=np.float64)
        
        # Transform to world coordinates
        return self.camera_rot.T @ camera_coordinates + self.camera_coor.flatten()

        

