from pathlib import Path
import time
import cv2
import numpy as np
import os
import uuid
import shutil

OUTPUT_PATH_DIR = os.path.join(os.getcwd(), "data", "output", "videos")


class VideoProcessor:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(VideoProcessor, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def sample_frames(
        self,
        video_path,
        interval_seconds=5,
        output_folder=None,
    ):
        """
        Sample frames from a video at specified time intervals.

        :param video_name: video file output name.
        :param interval_seconds: Time interval (in seconds) between sampled frames.
        :param output_folder: Folder where the sampled frames will be saved.
        :return: The output folder path where the frames are saved.
        """
        try:
            # Check if either video_path or video_bytes is provided
            if video_path is None:
                raise ValueError("Either video_path or video_bytes must be provided.")

            # Ensure the output folder exists
            output_folder = output_folder or OUTPUT_PATH_DIR
            video_name = os.path.basename(video_path)

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # Open the video file or video bytes
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                raise IOError("Error: Could not open video.")

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
            frame_interval = int(fps * interval_seconds)  # Number of frames to skip

            frame_count = 0
            saved_frame_count = 0
            output_paths = []
            base_directory = Path(f"{output_folder}/{video_name}")
            temp_directory = str(Path(f"{output_folder}/temp_{uuid.uuid4().hex}"))
            output_directory = str(Path(f"{output_folder}/{base_directory.stem}"))

            if os.path.exists(temp_directory):
                shutil.rmtree(str(temp_directory))

            if not os.path.exists(temp_directory):
                os.makedirs(temp_directory)

            print("Frame sampling ...")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # End of video

                if frame_count % frame_interval == 0:
                    # Save the frame
                    output_path = str(
                        Path(
                            f"{temp_directory}/frame_{saved_frame_count:04d}.jpg"
                        ).resolve()
                    )
                    cv2.imwrite(output_path, frame)

                    output_paths.append(output_path)
                    saved_frame_count += 1

                frame_count += 1

            cap.release()
            print("Frame sampling complete.")

            # Rename Directory
            if os.path.exists(output_directory):
                shutil.rmtree(str(output_directory))
            os.rename(str(temp_directory), str(output_directory))
            
            max_attempts = 5
            attempt = 0
            while attempt < max_attempts:
                if os.path.exists(output_directory):
                    break
                attempt += 1
                time.sleep(0.1)

            final_output_paths = [
                str(file) for file in Path(output_directory).iterdir() if file.is_file()
            ]
            print(f"Saved frame {saved_frame_count} to {output_directory}")
            return final_output_paths

        except Exception as e:
            print(f"An error occurred: {e}")
            return None
