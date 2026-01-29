import numpy as np
import cv2 as cv
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import robotic as ry

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
        for (frame, depth_frame) in self.camera.get_rgb_and_depth():
            # YOLO prediction
            results = self.predict_frame(frame, confidence=0.5)
            ball_list = self.process_results(results)
            for ball in ball_list:
                if self.display:
                    self.show_result(frame, ball)
                    # self.get_world_from_camera(ball, depth_frame)
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
    
    def show_result(self, frame, ball):
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

    def get_world_from_camera(self, ball:Component, depth_frame: np.ndarray) -> Optional[np.ndarray]:

        x1, y1, x2, y2 = ball.bbox
        image_points = np.array([
            [x1, y1],  # top-left
            [x2, y1],  # top-right
            [x2, y2],  # bottom-right
            [x1, y2]   # bottom-left
        ], dtype=np.float32)
        x_length = ball.bbox[2] - ball.bbox[0]
        y_length = ball.bbox[3] - ball.bbox[1]
        object_points = np.array([
            [-x_length/2,  y_length/2, 0],
            [ x_length/2,  y_length/2, 0],
            [ x_length/2, -y_length/2, 0],
            [-x_length/2, -y_length/2, 0]
        ], dtype=np.float32)

        T_world2cam = np.array([[-1, 0, 0, 0],[0, 1, 0, 0.45],[0, 0, -1, 2.30],[0, 0, 0, 1]]) 
        T_flipY = np.diag([1, -1, 1, 1])
        dist = np.zeros(5)
        _, rvec, tvec = cv.solvePnP(objectPoints=object_points, imagePoints=image_points, cameraMatrix=self.K, distCoeffs=dist)
        T_c_m = self.rvec_tvec_to_T(rvec, tvec)
        T_c_m_flip = T_flipY @ T_c_m
        T_w_m  = T_world2cam @ T_c_m_flip
        pose = self.matrix2vector(T_w_m)

        print(pose)
        

        

