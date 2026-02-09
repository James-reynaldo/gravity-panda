import cv2
import os
import logging
from pathlib import Path

from real_sense import RealSense

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')


class VideoStreamRecorder:
    """Records video stream from VideoStreamConsumer to file with auto-incremented filenames."""
    
    def __init__(self, output_dir='recordings', base_name='video', extension='mp4'):
        """
        Initialize the video recorder.
        
        Args:
            output_dir: Directory to save recordings
            base_name: Base name for video files
            extension: Video file extension (mp4, avi, etc.)
        """
        self.output_dir = Path(output_dir)
        self.base_name = base_name
        self.extension = extension
        self.logger = logging.getLogger("VideoStreamRecorder")
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_next_filename(self):
        """
        Generate next available filename in the output directory.
        Uses numbered sequence: video_001.mp4, video_002.mp4, etc.
        
        Returns:
            Path object for the next available filename
        """
        existing_files = list(self.output_dir.glob(f"{self.base_name}_*.{self.extension}"))
        
        if not existing_files:
            return self.output_dir / f"{self.base_name}_001.{self.extension}"
        
        # Extract numbers from existing files
        numbers = []
        for file in existing_files:
            try:
                # Extract number between base_name_ and .extension
                num_str = file.stem.replace(f"{self.base_name}_", "")
                numbers.append(int(num_str))
            except ValueError:
                continue
        
        # Get next number
        next_num = max(numbers) + 1 if numbers else 1
        return self.output_dir / f"{self.base_name}_{next_num:03d}.{self.extension}"
    
    def record(self, host='localhost', port=9999, fps=30.0, codec='mp4v', display=True):
        """
        Record video stream to file.
        
        Args:
            host: Server hostname
            port: Server port
            fps: Frames per second for output video
            codec: FourCC codec code ('mp4v' for MP4, 'XVID' for AVI, 'avc1' for H.264)
            display: Whether to display the video while recording
        """
        output_path = self.get_next_filename()
        self.logger.info(f"Recording will be saved to: {output_path}")
        self.logger.info("Press 'q' to stop recording")
        
        writer = None
        frame_count = 0
        
        try:
            with RealSense() as camera:
                camera.connect()
                camera.start_grabbing()

                while True:
                    frame, _ = camera.get_rgb_and_depth()
                    # Initialize video writer on first frame
                    if writer is None:
                        height, width = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        writer = cv2.VideoWriter(
                            str(output_path),
                            fourcc,
                            fps,
                            (width, height)
                        )
                        self.logger.info(f"Recording started: {width}x{height} @ {fps} fps")
                    
                    # Write frame to video file
                    writer.write(frame)
                    frame_count += 1
                    
                    # Display frame count every 30 frames
                    if frame_count % 30 == 0:
                        self.logger.info(f"Recorded {frame_count} frames")
        
        except KeyboardInterrupt:
            self.logger.info("\nRecording stopped by user")
        except Exception as e:
            self.logger.error(f"Error during recording: {e}")
        finally:
            # Release video writer
            if writer is not None:
                writer.release()
                self.logger.info(f"Recording saved: {output_path}")
                self.logger.info(f"Total frames recorded: {frame_count}")
                self.logger.info(f"Duration: {frame_count/fps:.2f} seconds")
            else:
                self.logger.warning("No frames were recorded")


def main():
    """Main function to run the video recorder."""
    # Create recorder instance
    recorder = VideoStreamRecorder(
        output_dir='recordings',
        base_name='video',
        extension='mp4'
    )
    
    # Start recording
    # Adjust host, port, and fps according to your setup
    recorder.record(
        host='localhost',
        port=9999,
        fps=30.0,
        codec='mp4v',  # Use 'XVID' for .avi, 'avc1' for H.264
        display=True
    )


if __name__ == "__main__":
    main()