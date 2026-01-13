import sys
import time
import threading
import cv2 as cv
import numpy as np
import robotic as ry
import matplotlib.pyplot as plt


class Camera():
    def __init__(self, bot) -> None:
        self.bot = bot
        
        self.camera_thread = None

    def start(self):
        self.camera_thread = threading.Thread(target=self.detector_loop, args=[], daemon=True)
        self.camera_thread.start()

    def shutdown(self):
        if self.camera_thread:
            self.camera_thread.join(timeout=1.0)

    def detector_loop(self):
        """
        Docstring for detector_loop
        """
        stop_detect = False

        while not stop_detect:
            pass

def process_frame(rgb, depth, K, camera_rot, camera_coor):
    """
    Process a single frame for ball detection
    
    Returns:
        Tuple of (best_ball_world_coor, best_circle, mask_blur)
    """
    mask_blur = filter_image_by_color(rgb)
    circles = circl_hough_transform(mask_blur)
    
    best_ball_world_coor = None
    best_circle = None
    
    if circles is not None and len(circles[0]) > 0:
        circles_array = circles[0]
        max_radius_idx = np.argmax(circles_array[:, 2])
        best_circle = np.uint16(np.around(circles_array[max_radius_idx]))
        
        best_ball_world_coor = get_world_from_camera(
            circle=best_circle, depth=depth, K=K,
            camera_rot=camera_rot, camera_coor=camera_coor
        )
 
    return best_ball_world_coor, best_circle, mask_blur

# Level 3: Simple detection stream
def detect_stream(image_generator, K, camera_rot, camera_coor, display=False):
    """Simple detection - yields (world_coor, circle)"""
    for frame_id, rgb, depth in image_generator:
        world_coor, circle, mask_blur = process_frame(rgb, depth, K, camera_rot, camera_coor)

        # Display logic if requested
        if display:
            display_circles = None
            if circle is not None:
                display_circles = np.array([[circle]])  # Shape [1, 1, 3]
            
            rgb_display = draw_detected_circles(rgb, display_circles)
            
            fig, axs = plt.subplots(1, 3, figsize=(10, 5))
            axs[0].imshow(rgb, cmap='gray')
            axs[0].set_title("Original")
            axs[0].axis('off')
            
            axs[1].imshow(mask_blur, cmap='gray')
            axs[1].set_title("Color Mask")
            axs[1].axis('off')
            
            axs[2].imshow(rgb_display)
            axs[2].set_title("Detected Circle(s)")
            axs[2].axis('off')
            
            plt.tight_layout()
            plt.show()
        
        yield world_coor, circle

def filter_image_by_color(rgb, color_lower: np.ndarray = np.array([90, 50, 50]), 
                          color_upper: np.ndarray = np.array([130, 255, 255]), 
                          blur_parameter = 5) -> np.ndarray:
    """
    lower_blue = np.array([90, 50, 50])    # Dark blue
    upper_blue = np.array([130, 255, 255]) # Light blue
    returns a binary mask where the blue colors are white and the rest is black
    """
    hsv = cv.cvtColor(rgb, cv.COLOR_RGB2HSV)
    mask = cv.inRange(hsv, color_lower, color_upper)
    mask_blur = cv.medianBlur(mask, blur_parameter)
    return mask_blur

def circl_hough_transform(mask: np.ndarray, dp: float = 1, minDist: float = 100,
                         param1: float = 100, param2: float = 15,
                         minRadius: int = 5, maxRadius: int = 30) -> np.ndarray:
    """
    Docstring for circl_hough_transform
    """
    circles = cv.HoughCircles(mask, cv.HOUGH_GRADIENT, dp, minDist,
                               param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    
    return circles
def draw_detected_circles(rgb: np.ndarray, circles: np.ndarray) -> np.ndarray:
    """
    Docstring for draw_detected_circles
    """
    output = rgb.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)
    return output

def get_world_from_camera(circle: tuple, depth: np.ndarray,
                          K: np.ndarray, camera_rot: np.ndarray, 
                          camera_coor: np.ndarray) -> np.ndarray:
    """
    Convert a single circle's camera coordinates to world coordinates.
    
    Args:
        circle: Tuple/list in format (center_x, center_y, radius)
        depth: Depth image/map (same resolution as RGB image)
        K: Camera intrinsic matrix (3x3)
        camera_rot: Camera rotation matrix (3x3) - camera to world
        camera_coor: Camera position in world coordinates (3,1) or (3,)
    
    Returns:
        World coordinates of the circle center (3,) or None if depth is 0
    """
    # Placeholder for actual implementation
    u,v = circle[0], circle[1]
    z = depth[v,u] # Convert mm to meters
    if z == 0:
        print("Depth is zero at the circle center; cannot compute world coordinates.")
        return None
    x = (u - K[0,2]) * z / K[0,0] # x = (u - cx) * cam_depth / fx
    y = (v - K[1,2]) * z / K[1,1] # y = (v - cy) * cam_depth / fy
    ball_cam_coor = np.array([x, y, z])
    ball_world_coor = camera_rot @ ball_cam_coor + camera_coor
    print("Ball world coordinates: ", ball_world_coor)

    return ball_world_coor


def camera_stream(bot: ry.BotOp, camera_name, max_frames=None, single_frame=False):
    """
    Generator that yields camera frames with optional limit
    
    Args:
        bot: BotOp instance
        camera_name: Name of the camera to use
        max_frames: Optional maximum number of frames to yield
    Yields:
        Tuple of (frame_id, rgb, depth)
    """
    if single_frame:
        rgb, depth, _ = bot.getImageDepthPcl(camera_name)
        yield 0, rgb, depth
        return
    else:
        frame_id = 0
        while True:
            if max_frames is not None and frame_id >= max_frames:
                break
                
            rgb, depth, _ = bot.getImageDepthPcl(camera_name)
            yield frame_id, rgb, depth
            frame_id += 1


def do_camera(args, bot, camera_pos, K, camera_rot, camera_coor, ball_position):
        stream = camera_stream(bot, camera_pos[args.input], single_frame=True)
        for result in detect_stream(stream, K, camera_rot, camera_coor, display=args.display):
            ball_position, _ = result
            if ball_position is None:
                print("No ball detected in the frame.")
                time.sleep(0.1)
        
        # if args.display:
        #     exit(0)


def main(argv):
    detector_loop()

if __name__ == "__main__":
    main(sys.argv[1:])