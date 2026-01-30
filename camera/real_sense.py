import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import cv2

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
        
    def connect(self, serial="819312071273"):
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
        while True:
            if not self.is_grabbing:
                raise Exception        
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
        
            if not color_frame:
                continue
            return color_frame, depth_frame                     


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
    
    def close(self):
        self.pipeline.stop()
    

    

    def plot_rgb_depth_with_boxes(self, rgb_image,
                                depth_image,
                                rgb_bboxes=None,
                                depth_bboxes=None,
                                figsize=(12, 8)):
        """
        Plots four subplots:
        1. Top-left: RGB image
        2. Top-right: RGB image with bounding boxes (None → no boxes)
        3. Bottom-left: Depth image
        4. Bottom-right: Depth image with bounding boxes (None → no boxes)

        Parameters
        ----------
        rgb_image : np.ndarray
            RGB image (H, W, 3)
        depth_image : np.ndarray
            Depth image (H, W)
        rgb_bboxes : list of tuples or None
            Each bbox as (x1, y1, x2, y2). None → no boxes.
        depth_bboxes : list of tuples or None
            Same format as rgb_bboxes.
        figsize : tuple
            Figure size.
        """

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # --- 1. RGB image ---
        axes[0, 0].imshow(rgb_image)
        axes[0, 0].set_title("RGB Image")
        axes[0, 0].axis("off")

        # --- 2. RGB image with bounding boxes ---
        axes[0, 1].imshow(rgb_image)
        axes[0, 1].set_title("RGB + Boxes")
        axes[0, 1].axis("off")

        if rgb_bboxes is not None:
            for (x1, y1, x2, y2) in rgb_bboxes:
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                    fill=False, edgecolor='green', linewidth=2)
                axes[0, 1].add_patch(rect)

        # --- 3. Depth image ---
        axes[1, 0].imshow(depth_image, cmap='gray')
        axes[1, 0].set_title("Depth Image")
        axes[1, 0].axis("off")

        # --- 4. Depth image with bounding boxes ---
        axes[1, 1].imshow(depth_image, cmap='gray')
        axes[1, 1].set_title("Depth + Boxes")
        axes[1, 1].axis("off")

        if depth_bboxes is not None:
            for (x1, y1, x2, y2) in depth_bboxes:
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                    fill=False, edgecolor='yellow', linewidth=2)
                axes[1, 1].add_patch(rect)

        plt.tight_layout()
        plt.show()

    def show_grab_results(self, depth_gray:bool = False, depth_color:bool = False):    
        """Showing the RGB frames of the current frame grap. To show the depth frame: set depth flags to show in gray scale or with color map."""
        while True:
            rgb,depth = self.get_rgb_and_depth()
            self.show_rgb_frame(rgb)
            if depth_gray:
                self.show_depth_gray(depth)
            if depth_color:
                self.show_depth_colormap(depth)

            if cv2.waitKey(1) & 0xFF == 27:
                break

            

    def show_rgb_frame(self, rgb_frame):
        cv2.imshow("RealSense RGB", rgb_frame)

    def show_depth_gray(self, depth_frame):
        """
        depth_frame: 16-bit depth image from Intel RealSense
        """
        # Normalize depth to 0–255
        depth_norm = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = depth_norm.astype(np.uint8)

        cv2.imshow("RealSense Depth (Gray)", depth_norm)


    def show_depth_colormap(self, depth_frame):
        """
        depth_frame: 16-bit depth image from Intel RealSense
        """
        # Normalize depth to 0–255
        depth_norm = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = depth_norm.astype(np.uint8)

        # Apply a colormap (JET is common for depth)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

        cv2.imshow("RealSense Depth (Colormap)", depth_color)


def main():
    """Example how to show the grabbed frame of the RealSense."""
    print("Starting RealSense.")
    camera = RealSense()
    camera.connect()
    camera.start_grabbing()

    print("Showing frames:")
    camera.show_grab_results(depth_color=True)

    camera.close()
    print("Program exited cleanly.")

if __name__ == "__main__":
    main()