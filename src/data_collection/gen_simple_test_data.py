import random
from typing import List, Any, Tuple
import numpy as np
import numpy.typing as npt
import cv2
import gymnasium
from gymnasium import spaces

class SimpleObject:
    def __init__(self, x: int, y: int, w: int, h: int, lx: int, ly: int) -> None:
        self.xywh = (x,y,w,h)
        self.xy = (x,y)
        self.last_xy = (lx,ly)
        self.category = "Test"

class SimpleTestData(gymnasium.Env):
    def __init__(self) -> None:
        super().__init__()
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(128, 128, 3), dtype=np.uint8
        )
        self.all_objects: List[List[SimpleObject]]= []
        self.frames: List[np.ndarray] = self.gen_frames()
        self.frame_idx = 0
        self.state = 0
        self.objects: List[SimpleObject] = []

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        frame = self.frames[self.frame_idx]
        self.objects = self.all_objects[self.frame_idx]
        self.frame_idx += 1
        done = self.frame_idx == len(self.frames)
        # Return the frame, reward, done, and info
        return frame, 0.0, done, done, {}

    def render(self,
               mode: str = "human" # pylint: disable=unused-argument
               ) -> None:
        return

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None,) -> Tuple[npt.NDArray, Any]:
        super().reset(seed=seed)
        self.frame_idx = 0
        return self.frames[self.frame_idx], None

    def gen_frames(self) -> list[np.ndarray]:
        # Set the dimensions of the images
        image_width = 128
        image_height = 128

        # Set the number of frames in the sequence
        num_frames = 10

        # Create an empty list to store the frames
        frames = []

        objects = [[random.randint(0, 30), random.randint(0, 30)], [random.randint(100, 120), random.randint(50, 80)]]
        speeds = [[random.randint(0, 3), random.randint(0, 3)], [ random.randint(-3, 0), random.randint(-3, 0)]]

        # Generate each frame
        for i in range(num_frames):
            # Create a blank image
            image = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255

            # Calculate the position of the circle
            for i in range(len(objects)):
                x,y = objects[i]
                nx, ny = x + speeds[i][0], y + speeds[i][1]
                if nx < 0 or nx > image_width:
                    speeds[i][0] *= -1
                if ny < 0 or ny > image_height:
                    speeds[i][1] *= -1
                circle_x = nx
                circle_y = ny
                self.all_objects.append([SimpleObject(circle_x - 10, circle_y - 10, 10, 10, x, y)])

                # Draw the circle on the image
                cv2.circle(image, (circle_x, circle_y), 10, 0, -1)  # type: ignore
                objects[i] = [nx, ny]

            # Add the frame to the list
            frames.append(image)
        return frames
