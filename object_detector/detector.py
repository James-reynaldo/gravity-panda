import sys
import cv2 as cv
import numpy as np

def detector_loop():
    """
    Docstring for detector_loop
    """
    stop_detect = False

    while not stop_detect:
        pass

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


def main(argv):
    detector_loop()

if __name__ == "__main__":
    main(sys.argv[1:])