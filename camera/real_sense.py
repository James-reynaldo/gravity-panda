import pyrealsense2 as rs
import numpy as np

class DetectedObject():
    def __init__(self, name :str=None, bbox :np.ndarray=np.array([None, None, None, None])):
        self.name :str = name
        self.bbox: np.ndarray = bbox
        self.center: list = self.get_bbox_center(self.bbox.tolist())
        self.q_pose = [None, None, None, None, None, None, None]

    def get_bbox_center(self, bbox) -> list:
        if bbox is None or any(v is None for v in bbox):
            return [None, None]
        
        x_min, y_min, x_max, y_max = bbox
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        return [x_center, y_center]
    
    def print_obj(self):
        print()
        print(f"{self.name.capitalize()}")
        print(f"- BBox: {self.bbox}")
        print(f"- Center: {self.center}")
        print(f"- Pose (quat.): {self.q_pose}")

class RealSense():
    """Class to control RealSenseCamera"""
    def __init__(self) -> None:
        self.pipeline = None
        self.config = None
        self.profile = None
        self.serial = None
        self.is_grabbing = False

        # Initialize camera parameter with default values
        self.K = np.array([[1.0, 0,  1.0],
                    [0,  1.0, 1.0],
                    [0,  0,  1.0]]) 
        self.coordinates = np.array([0.0, 0.0, 0.0])
        self.rotation = np.eye(3)
        
    def connect(self, serial=None):
        print("[RealSense] Connecting device...")
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if serial:
            self.serial = serial
            self.config.enable_device(self.serial)

    def start_grabbing(self, serial=None, width=1280, height=720, fps=30):
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.profile = self.pipeline.start(self.config)
        # Set up alignment so depth matches color frame
        align_to = rs.stream.color
        align = rs.align(align_to)
        self.is_grabbing = True

    def acquire_images(self, serial=None):
        try:
            while True:
                if not self.is_grabbing:
                    raise Exception        
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
            
                if not color_frame:
                    continue
                return color_frame, depth_frame                     
        finally:
            self.pipeline.stop()

        return None, None

    def get_rgb_and_depth(self):
        color_frame, depth_frame = self.acquire_images()
        rgb = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())  
        return rgb, depth   
        
    def get_color_intrinsics(self):
        color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        # intr: width, height, ppx, ppy, fx, fy, model, coeffs (5)
        K = np.array([[intr.fx, 0, intr.ppx],
                    [0, intr.fy, intr.ppy],
                    [0, 0, 1]], dtype=np.float64)
        # Map RS distortion to OpenCV k1,k2,p1,p2,k3 (Brown-Conrady)
        if intr.model in (rs.distortion.brown_conrady, rs.distortion.inverse_brown_conrady):
            coeffs = np.array(intr.coeffs[:5], dtype=np.float64)
        else:
            coeffs = np.zeros(5, dtype=np.float64)
        return K, coeffs, intr.width, intr.height
    
    def process_yolo_result(self, results):
        detection_list = []
        for result in results:
                for box in result.boxes:
                    name = f"{result.names[int(box.cls[0])]}"
                    bbox = box.xyxy[0]
                    detection_list.append(DetectedObject(name, bbox))
        return detection_list
    