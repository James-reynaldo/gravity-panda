import argparse
import sys
import time
import threading
import cv2 as cv
import numpy as np
import robotic as ry
import matplotlib.pyplot as plt
import os
import queue
from matplotlib.animation import FuncAnimation
import collections
from camera.bot_camera import RobotCamera

class Detector():
    def __init__(self, camera:RobotCamera, kernel_size=(5,5), display:bool=False) -> None:
        self.camera = camera
        self.kernel_size = kernel_size
        self.min_area = 100
        # Thread safety
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Thread-safe queues for communication
        self.plot_queue = queue.Queue(maxsize=10)  # Limit queue size to prevent memory issues
        self.result_queue = queue.Queue()
        self.display = display
        
        # Matplotlib figure (created once in main thread)
        self.fig = None
        self.ax = None
        self.anim = None
        self.img_display = None

    # ===== Frame Processing =====

    def filter_image(self, rgb, color_lower: np.ndarray = np.array([90, 50, 50]), 
                            color_upper: np.ndarray = np.array([130, 255, 170]), 
                            blur_parameter = 5) -> np.ndarray:
        """
        Filter the image to isolate the ball based on color.
        """
        # Color Filtering + Blurring
        hsv = cv.cvtColor(rgb, cv.COLOR_RGB2HSV)
        mask = cv.inRange(hsv, color_lower, color_upper)
        mask_blur = cv.medianBlur(mask, blur_parameter)

        # Seperate touching objects
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, self.kernel_size)
        mask_sep = cv.morphologyEx(mask_blur, cv.MORPH_OPEN, kernel)
        return mask_sep
    
    def find_circle_by_contours(self, mask: np.ndarray) -> list:
        """
        Find circles in a mask using contour analysis.
        """
        cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        print("contour areas:", [cv.contourArea(c) for c in cnts])
        # Keep only large blobs
        cnts = [c for c in cnts if cv.contourArea(c) > self.min_area]

        print("Found {} contours after area filtering.".format(len(cnts)))
        candidates = []

        for c in cnts:
            area = cv.contourArea(c)


            perimeter = cv.arcLength(c, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)

            # Ball ≈ circular, screen ≈ not
            if circularity > 0.4:
                candidates.append(c)
        return candidates
    
    def get_world_from_camera(self, circle: tuple, depth: np.ndarray,
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
        
        u,v = circle[0], circle[1]
        if depth is not None:
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


    def detect(self, rgb, depth):
        """
        Process a single frame for ball detection
        
        Returns:
            Tuple of (best_ball_world_coor, best_circle, mask_blur)
        """
        mask_blur = self.filter_image(rgb)
        candidates = self.find_circle_by_contours(mask_blur)
        if not candidates:
            print("NO BALL candidates found.")
            return None, None, mask_blur
        
        c = max(candidates, key=cv.contourArea)

        best_circle = None
        (x, y), r = cv.minEnclosingCircle(c)

        best_circle = (int(x), int(y), int(r))
        best_ball_world_coor = self.get_world_from_camera(
            circle=best_circle, depth=depth, K=self.camera.K,
            camera_rot=self.camera.rotation, camera_coor=self.camera.coordinates
        )
    
        return best_ball_world_coor, best_circle, mask_blur

    def process_frame(self, frame):
        rgb, depth = frame
        world_coor, circle, mask_blur = self.detect(rgb, depth)
        if world_coor is None:
            print("No world coordinates, skip plot.")
            return None, mask_blur, rgb
        
        if self.display:
            if circle is not None:
                display_circles = np.array([[circle]])  # Shape [1, 1, 3]
                rgb = self.draw_detected_circles(rgb, display_circles)
        return world_coor, mask_blur, rgb

    def run(self):
        while not self._stop_event.is_set():
            dframe = self.camera.grab_once()
            result = self.process_frame(dframe)
            self.ball_position, mask_blur, rgb_display = result
            mask_blur_color = cv.cvtColor(mask_blur, cv.COLOR_GRAY2RGB)
            
            if self.ball_position is None:
                print("NO BALL detected.")
                time.sleep(0.1)
                continue

            try:
                self.plot_queue.put_nowait(rgb_display)
                self.plot_queue.put_nowait(mask_blur_color)
            except queue.Full:
                print("Queue is full, clearing and adding again.")
                self.clear_queue(self.plot_queue)
                self.plot_queue.put_nowait(rgb_display)
                self.plot_queue.put_nowait(mask_blur_color)

            self.update_plot(-1)

    # ===== Image display =====

    def setup_plot(self):
        """Setup plot once in main thread"""
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 8))
        self.ax.set_title("Camera Feed")
        self.ax.axis('off')

        # Initial empty image
        self.img_display = self.ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        self.anim = FuncAnimation(self.fig, self.update_plot, interval=50, blit=False, cache_frame_data=False)

    def update_plot(self, frame_num):
        """Called by matplotlib animation in main thread"""
        try:
            # Get latest frame from queue (non-blocking)
            try:
                rgb_display = self.plot_queue.get_nowait()
                
                # Update plot
                if rgb_display is not None:
                    self.img_display.set_data(rgb_display)
                    
                # Clear queue to only show latest frame
                self.clear_queue(self.plot_queue)
                        
            except queue.Empty:
                pass  # No new frame, keep showing old one
        except Exception as e:
            print(f"Error updating display: {e}")

        return [self.img_display]

    
    def render_plot(self):
        """For compatibility - just pause briefly"""
        if self.display and self.fig:
            plt.pause(0.001)


    def draw_detected_circles(self, rgb: np.ndarray, circles: np.ndarray) -> np.ndarray:
        """
        rgb: Original RGB image
        circles: Detected circles (shape [1, N, 3] or None)
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
    

    # ===== Lifetime Management =====

    def clear_queue(self, given_queue):
        """Clear queue."""
        while not given_queue.empty():
            try:
                given_queue.get_nowait()
            except given_queue.Empty:
                break

    def start(self):
        if not self.plot_queue.empty():
            self.clear_queue(self.plot_queue)

        self.camera_thread = threading.Thread(target=self.run, args=[self.ball_position], daemon=True)
        self.camera_thread.start()
        
        if self.display:
            self.setup_plot()

    def shutdown(self):
        self._stop_event.set()
        if self.display or self.fig:
            plt.close(self.fig)
        self.clear_queue(self.plot_queue)
        print("Detector shutdown complete.")


def main():
    camera = Camera(display=True, display_target="mask")
    camera.start()
    try:
        if camera.display:
            # Matplotlib will handle updates via animation
            plt.show(block=True)  # This blocks until window is closed
        else:
            # If not displaying, just keep running
            while True:
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        camera.shutdown()
        print("Program exited cleanly.")


if __name__ == "__main__":
    main()
    