import numpy as np
import cv2
import gymnasium
from gymnasium import spaces


class SimpleTestData(gymnasium.Env):
    def __init__(self):
        super(SimpleTestData, self).__init__()
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(128, 128, 3), dtype=np.uint8
        )
        self.frames = self.gen_frames()
        self.frame_idx = 0
        self.state = 0

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        frame = self.frames[self.frame_idx]
        self.frame_idx += 1
        done = self.frame_idx == len(self.frames)
        # Return the frame, reward, done, and info
        return frame, 0.0, done, done, {}

    def reset(self) -> np.ndarray:
        self.frame_idx = 0
        return self.frames[self.frame_idx]

    def gen_frames(self) -> list[np.ndarray]:
        # Set the dimensions of the images
        image_width = 128
        image_height = 128

        # Set the number of frames in the sequence
        num_frames = 10

        # Create an empty list to store the frames
        frames = []

        # Generate each frame
        for i in range(num_frames):
            # Create a blank image
            image = np.ones((image_height, image_width), dtype=np.uint8) * 255

            # Calculate the position of the circle
            circle_x = int(i * (image_width / num_frames))
            circle_y = int(i * (image_height / num_frames))

            # Draw the circle on the image
            cv2.circle(image, (circle_x, circle_y), 10, 0, -1)

            # Add the frame to the list
            frames.append(image)
        return frames
