import numpy as np
import cv2 as cv
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import robotic as ry
import threading
from object_detector.aruco_detector import detect_aruco_markers

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
        self.current_ball_coor = None
    
        self._lock = threading.Lock()


    def run(self):
        for (frame, depth_frame) in self.camera.get_rgb_and_depth():
            # YOLO prediction
            results = self.predict_frame(frame, confidence=0.3)
            ball=  self.process_results(results)
            if self.display:
                self.show_result(frame, ball)
                if cv.waitKey(1) & 0xFF == 27:
                    break
            
            # K, dist, W, H = self.camera.get_color_intrinsics()
            # detect_aruco_markers(frame, K, dist, W, H)
            
            with self._lock:
                if ball:
                    self.current_ball_coor = self.get_world_from_camera(ball, depth_frame, self.K, self.camera_rot, self.camera_coor)
                else:
                    self.current_ball_coor = None
                    
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
            return None

        for result in results:
            names = result.names  # FIX: cache names once
            for box in result.boxes:
                cls_id = int(box.cls[0])
                component = Component(names[cls_id], float(box.conf[0]), box.xyxy[0].tolist())
                if component.name == "ball":
                    return component
        return None
    
    def show_result(self, frame, ball= None):
        if ball is not None:
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


    def rt_to_T(self, R, t):
        T = np.eye(4, dtype=float)
        T[:3,:3] = R
        T[:3, 3] = t.reshape(3)
        return T

    def rvec_tvec_to_T(self, rvec, tvec):
        R, _ = cv.Rodrigues(rvec)
        return self.rt_to_T(R, tvec)     
    
    def matrix2vector(self, T):
        vec = np.zeros(7) 
        R = T[:3, :3]
        quat = ry.Quaternion().setMatrix(R).asArr() 
        vec[:3] = T[:3, 3]
        vec[3:] = quat
        return vec  

    def get_world_from_camera(self, ball:Component, depth: np.ndarray,
                            K: np.ndarray, camera_rot: np.ndarray, 
                            camera_coor: np.ndarray) -> np.ndarray:

        # Placeholder for actual implementation
        u = int(ball.center[0])
        v = int(ball.center[1])
        if depth is not None and not any(i is None for i in [u,v]):
            z = depth[v,u] # Convert mm to meters
        else:
            z = 1.0  # Default depth if depth image is not available
        if z == 0:
            print("Depth is zero at the circle center; cannot compute world coordinates.")
            return None
        x = (u - K[0,2]) * z / K[0,0] # x = (u - cx) * cam_depth / fx
        y = (v - K[1,2]) * z / K[1,1] # y = (v - cy) * cam_depth / fy
        ball_cam_coor = np.array([x, y, z])
        ball_world_coor = camera_rot @ ball_cam_coor + camera_coor
        print("Ball world coordinates: ", ball_world_coor)

        return ball_world_coor
    

