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


class Camera():
    def __init__(self, C:ry.Config= None ,input:str= None, bot: ry.BotOp= None, 
                 camera_pos:dict= None, display = False, max_frames:int= None, 
                 display_target = "rgb", kernel_size = (5, 5), min_area = 100,
                 threshold=0.6, color_lower=[10, 80, 40], color_upper = [30, 255, 255]) -> None:
        self.bot = bot
        self.C = C
        self.input = input
        self.camera_pos = camera_pos
        self.display = display
        self.max_frames = max_frames
        self.display_target = display_target # rgb or mask
        self.kernel_size = kernel_size
        self.min_area = min_area
        self.threshold = threshold
        self.color_lower = color_lower
        self.color_upper = color_upper
        
        self.camera_thread = None
        self.ball_position = None
        self.K = None
        self.camera_coor = None
        self.camera_rot = None
        self.ball_position = None
        self.camera_thread = None        

        # Thread-safe queues for communication
        self.plot_queue = queue.Queue(maxsize=10)  # Limit queue size to prevent memory issues
        self.result_queue = queue.Queue()
        self.running = False

        
        # Matplotlib figure (created once in main thread)
        self.fig = None
        self.ax = None
        self.anim = None

    def start(self):
        if self.bot is not None and self.input is not None and self.camera_pos is not None:
            fxycxy = self.bot.getCameraFxycxy(self.camera_pos[self.input]) 
            fx, fy, cx, cy = fxycxy
            self.K = np.array([[fx, 0,  cx],
                        [0,  fy, cy],
                        [0,  0,  1.0]]) 
        else:
            print("Using default camera intrinsics.")
            self.K = np.array([[1.0, 0,  1.0],
                        [0,  1.0, 1.0],
                        [0,  0,  1.0]]) 
        if self.C is not None:
            self.camera_coor = self.C.getFrame(self.camera_pos[self.input]).getPosition()
            self.camera_rot = self.C.getFrame(self.camera_pos[self.input]).getRotationMatrix()
        else:
            print("Using default camera extrinsics.")
            self.camera_coor = np.array([0.0, 0.0, 0.0])
            self.camera_rot = np.eye(3)
        # Add a stop flag for graceful shutdown
        self.running = True

        # Clear queues
        while not self.plot_queue.empty():
            try:
                self.plot_queue.get_nowait()
            except queue.Empty:
                break

        self.camera_thread = threading.Thread(target=self.detector_loop, args=[self.ball_position], daemon=True)
        self.camera_thread.start()
        
        if self.display:
            self.setup_plot()
  

    def setup_plot(self):
        """Setup plot once in main thread"""
        if self.display:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 8))
            self.ax.set_title("Camera Feed")
            self.ax.axis('off')
        
            # Initial empty image
            self.img_display = self.ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
            
            # Use matplotlib's animation for smooth updates
            self.anim = FuncAnimation(self.fig, self.update_plot, interval=50, blit=False, cache_frame_data=False)

    def detector_loop(self, ball_position):
        try:
            stream = self.camera_stream()
            for result in self.detect_stream(stream):
                # Check if we should stop
                if not self.running:
                    break
                self.ball_position, mask_blur, rgb_display = result
                # print("shape of rgb_display:", None if rgb_display is None else rgb_display.shape)
                # print("shape of mask_blur:", mask_blur.shape)
                # print("range of mask blur:", np.min(mask_blur), np.max(mask_blur))
                print(f"Non-zero pixels: {np.count_nonzero(mask_blur)} / {mask_blur.size}")
                # print(f"Mean value: {np.mean(mask_blur):.2f}")
                mask_blur_color = cv.cvtColor(mask_blur, cv.COLOR_GRAY2RGB)
                if self.ball_position is None:
                    print("NO BALL detected in the frame.")
                    time.sleep(0.1)
                
                if self.display and rgb_display is not None:
                    try:  
                    # Make a copy to avoid thread issues
                        # rgb_disp_copy = rgb_display if rgb_display is not None else None
                        ####### EDITED ############
                        # rgb_disp_copy = mask_blur.copy()
                        ####### EDITED ############
                        if(self.display_target == "rgb"):
                            self.plot_queue.put_nowait(rgb_display)
                        elif(self.display_target == "mask"):
                            # self.plot_queue.put_nowait(rgb_disp_copy)
                            self.plot_queue.put_nowait(mask_blur_color)

                    except queue.Full:
                        # Drop old frame if queue is full
                        try:
                            self.plot_queue.get_nowait()  # Remove oldest
                            # self.plot_queue.put_nowait(rgb_disp_copy)
                            self.plot_queue.put_nowait(mask_blur_color)
                        except queue.Empty:
                            pass        

        except Exception as e:
            print(f"Error in detector loop: {e}")
        finally:
            print("Detector loop has stopped.")
                # if args.display:
                #     exit(0)
    
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
                while not self.plot_queue.empty():
                    try:
                        self.plot_queue.get_nowait()
                    except queue.Empty:
                        break
                        
            except queue.Empty:
                pass  # No new frame, keep showing old one
        except Exception as e:
            print(f"Error updating display: {e}")

        return [self.img_display]

    
    def render_plot(self):
        """For compatibility - just pause briefly"""
        if self.display and self.fig:
            plt.pause(0.001)

    def shutdown(self):
        # Set flag to stop the loop
        print("Shutting down camera...")
        self.running = False
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2.0)
        
        if self.display and self.fig:
            plt.close(self.fig)
        
        # Clear queues
        while not self.plot_queue.empty():
            try:
                self.plot_queue.get_nowait()
            except queue.Empty:
                break
        
        print("Camera shutdown complete")


    def process_frame(self, rgb, depth):
        """
        Process a single frame for ball detection
        
        Returns:
            Tuple of (best_ball_world_coor, best_circle, mask_blur)
        """
        mask_blur = self.filter_image(rgb)
        # circles = self.circl_hough_transform(mask_blur)
        candidates = self.find_circl_by_contours(mask_blur.copy())
        if not candidates:
            print("NO BALL candidates found.")
            return None, None, mask_blur
        else:
            c = max(candidates, key=cv.contourArea)

            best_circle = None
            (x, y), r = cv.minEnclosingCircle(c)

            best_circle = (int(x), int(y), int(r))
            best_ball_world_coor = self.get_world_from_camera(
                    circle=best_circle, depth=depth, K=self.K,
                    camera_rot=self.camera_rot, camera_coor=self.camera_coor
                )

            
            # if circles is not None and len(circles[0]) > 0:
            #     circles_array = circles[0]
            #     max_radius_idx = np.argmax(circles_array[:, 2])
            #     best_circle = np.uint16(np.around(circles_array[max_radius_idx]))
                
            #     best_ball_world_coor = self.get_world_from_camera(
            #         circle=best_circle, depth=depth, K=self.K,
            #         camera_rot=self.camera_rot, camera_coor=self.camera_coor
            #     )
        
            return best_ball_world_coor, best_circle, mask_blur

    # Level 3: Simple detection stream
    def detect_stream(self, image_generator):
        """Simple detection - yields (world_coor, circle)"""
        for _, rgb, depth in image_generator:
            world_coor, circle, mask_blur = self.process_frame(rgb, depth)
            if world_coor is None:
                print("world_coor is None, skipping display drawing.")
                rgb_display = rgb.copy()
                yield None, mask_blur, rgb_display
                continue
            # Display logic if requested
            if self.display:
                rgb_display = rgb.copy()
                if circle is not None:
                    display_circles = np.array([[circle]])  # Shape [1, 1, 3]
                    rgb_display = self.draw_detected_circles(rgb, display_circles)
            yield world_coor, mask_blur, rgb_display

    def filter_image(self, rgb, 
                            blur_parameter = 5) -> np.ndarray:
        """
        Docstring for filter_image
        """
        # Color Filtering + Blurring
        hsv = cv.cvtColor(rgb, cv.COLOR_RGB2HSV)
        mask = cv.inRange(hsv, self.color_lower, self.color_upper)
        mask_blur = cv.medianBlur(mask, blur_parameter)

        # Seperate touching objects
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, self.kernel_size)
        mask_sep = cv.morphologyEx(mask_blur, cv.MORPH_OPEN, kernel)
        return mask_sep

    def circl_hough_transform(self, mask: np.ndarray, dp: float = 1, minDist: float = 100,
                            param1: float = 100, param2: float = 15,
                            minRadius: int = 5, maxRadius: int = 100) -> np.ndarray:
        """
        Docstring for circl_hough_transform
        """
        circles = cv.HoughCircles(mask, cv.HOUGH_GRADIENT, dp, minDist,
                                param1=param1, param2=param2,
                                minRadius=minRadius, maxRadius=maxRadius)
        
        return circles

    def find_circl_by_contours(self, mask: np.ndarray) -> list:
        """
        Docstring for find_circl_contours
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
            if circularity > self.threshold:
                candidates.append(c)
        return candidates

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
        # Placeholder for actual implementation
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


    def camera_stream(self):
        """
        Generator that yields camera frames with optional limit
        
        Args:
            bot: BotOp instance
            camera_name: Name of the camera to use
            max_frames: Optional maximum number of frames to yield
        Yields:
            Tuple of (frame_id, rgb, depth)
        """
        if self.bot is not None and self.camera_pos[self.input] is not None:
            frame_id = 0
            while True:
                if self.max_frames is not None and frame_id >= self.max_frames:
                    break
                rgb, depth, _ = self.bot.getImageDepthPcl(self.camera_pos[self.input])
                yield frame_id, rgb, depth
                frame_id += 1
        else:
            for entry in os.scandir("./object_detector/images"):
                if entry.name.endswith(".jpg") or entry.name.endswith(".png"):
                    ## print full path: entry.path
                    print("Reading image: ", entry.name)
                    frame_id = entry.name.split("_")[0]
                    rgb = cv.imread(entry.path)
                    rgb = cv.cvtColor(rgb, cv.COLOR_BGR2RGB)
                    depth = None # Placeholder depth
                    yield frame_id, rgb, depth

def circl_hough_transform( mask: np.ndarray, dp: float = 1.2, minDist: float = 100,
                            param1: float = 100, param2: float = 20,
                            minRadius: int = 30, maxRadius: int = 500) -> np.ndarray:
        """
        Docstring for circl_hough_transform
        """
        circles = cv.HoughCircles(mask, cv.HOUGH_GRADIENT, dp, minDist,
                                param1=param1, param2=param2,
                                minRadius=minRadius, maxRadius=maxRadius)
        return circles
def main():
    # for entry in os.scandir("./object_detector/images"):
    #     if entry.name.endswith(".jpg") or entry.name.endswith(".png"):
    #         ## print full path: entry.path
    #         print("Reading image: ", entry.name)
    #         frame_id = entry.name.split("_")[0]
    #         rgb_old = cv.imread(entry.path)
    #         rgb = cv.cvtColor(rgb_old, cv.COLOR_BGR2RGB)
    #         depth = None # Placeholder depth
    #         # yield frame_id, rgb, depth
    #         # Convert RGB to HSV (better for color segmentation)
    #         hsv = cv.cvtColor(rgb, cv.COLOR_RGB2HSV)

    #         # Adjust these values based on your specific blue
    #         lower_blue = np.array([90, 50, 50])    # Dark blue
    #         upper_blue = np.array([130, 255, 170]) # Light blue
            
    #         # Create mask for blue regions
    #         blue_mask = cv.inRange(hsv, lower_blue, upper_blue)
    #         # Bitwise-AND mask and original imageE
    #         rgb_display = rgb.copy()
    #         rgb_display2 = rgb.copy()
    #         mask_blur = cv.medianBlur(blue_mask, 5)

    #         # circles = circl_hough_transform(mask_blur)

    #         # --- SEPARATE TOUCHING OBJECTS (Solution 1) ---
    #         kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (30, 30))
    #         mask_sep = cv.morphologyEx(mask_blur, cv.MORPH_OPEN, kernel)

    #         cnts, _ = cv.findContours(mask_sep, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #         # Keep only large blobs
    #         cnts = [c for c in cnts if cv.contourArea(c) > 500]

    #         print("Found {} contours after area filtering.".format(len(cnts)))

    #         #draw contours with different color
    #         for i, c in enumerate(cnts):
    #             color = (255 - i*50, i*50, 0)
    #             cv.drawContours(rgb_display2, [c], -1, color, 25)

    #         candidates = []

    #         for c in cnts:
    #             area = cv.contourArea(c)


    #             perimeter = cv.arcLength(c, True)
    #             if perimeter == 0:
    #                 continue

    #             circularity = 4 * np.pi * area / (perimeter * perimeter)

    #             # Ball ≈ circular, screen ≈ not
    #             if circularity > 0.40:
    #                 candidates.append(c)
            
    #         if not candidates:
    #             print("NO BALL candidates found.")
    #             rgb_display = rgb.copy()
                
    #         else:
    #             c = max(candidates, key=cv.contourArea)

    #             (x, y), r = cv.minEnclosingCircle(c)

    #             #plot circle
    #             cv.circle(rgb_display, (int(x), int(y)), int(r), (0, 255, 0), 25)
    #             cv.circle(rgb_display, (int(x), int(y)), 2, (0, 0, 255), 50)

    #         # if circles is not None and len(circles[0]) > 0:
    #         #     circles_array = circles[0]
    #         #     max_radius_idx = np.argmax(circles_array[:, 2])
    #         #     best_circle = np.uint16(np.around(circles_array[max_radius_idx]))
    #         #     print(f"best radius {best_circle[2]}")
    #         #     # draw the outer circle
    #         #     cv.circle(rgb_display, (best_circle[0], best_circle[1]), best_circle[2], (0, 255, 0), 25)
    #         #     # draw the center of the circle
    #         #     cv.circle(rgb_display, (best_circle[0], best_circle[1]), 2, (0, 0, 255), 50)
    #         # else:
    #         #     print("No circles detected.")
    #         ##plotting
    #         # Setup plot
    #         fig, ax = plt.subplots(1, 5, figsize=(10, 5))
    #         ax[0].imshow(rgb, cmap='gray')
    #         ax[0].set_title("Original")
    #         ax[0].axis('off')
    #         ax[1].imshow(mask_blur, cmap='gray')
    #         ax[1].set_title("Blurred Mask")      
    #         ax[1].axis('off')
    #         ax[2].imshow(mask_sep, cmap='gray')
    #         ax[2].set_title("Mask Sep")      
    #         ax[2].axis('off')
    #         ax[3].imshow(rgb_display2)
    #         ax[3].set_title("Contours")
    #         ax[3].axis('off')
    #         ax[4].imshow(rgb_display)
    #         ax[4].set_title("Detected Circle(s)")
    #         ax[4].axis('off')
    #         plt.show()

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
    