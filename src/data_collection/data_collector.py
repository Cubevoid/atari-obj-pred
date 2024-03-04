from typing import List, Tuple
import os
import numpy as np
import numpy.typing as npt
from ocatari.core import OCAtari
from ocatari.utils import load_agent
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch # type: ignore
import tqdm
import wandb

class DataCollector:
    def __init__(self, game: str, num_samples: int) -> None:
        self.game = game
        self.env = OCAtari(game, mode="revised", hud=True, obs_mode='dqn')
        self.agent = load_agent(f"./models/dqn_{game}.gz", self.env.action_space.n)
        self.num_samples = num_samples

        self.dataset_path = f"./data/{game}/"
        os.makedirs(self.dataset_path, exist_ok=True)

        self.curr_episode_id = 0
        self.determine_next_episode()
        self.collected_data = self.get_collected_data()
        self.episode_frames: List[npt.NDArray] = []
        self.episode_object_types : List[List[str]] = []
        self.episode_object_bounding_boxes : List[List[Tuple[int, int, int, int]]]= []
        self.episode_detected_masks : List[List[npt.NDArray]] = []
        self.episode_actions : List[int] = []

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry["vit_b"](checkpoint="./models/sam_vit_b_01ec64.pth").to(self.device)
        sam = torch.compile(sam)
        self.generator = SamAutomaticMaskGenerator(sam)

    def collect_data(self) -> None:
        """
        Collects data from the environment for a given number of steps.
        """
        progress_bar = tqdm.tqdm(total=self.num_samples)
        progress_bar.update(self.collected_data)
        obs, _ = self.env.reset()
        while self.collected_data < self.num_samples:
            action = self.agent.draw_action(self.env.dqn_obs)
            obs, _, terminated, truncated, _ = self.env.step(action)
            self.episode_frames.append(obs)
            self.episode_object_types.append([])
            self.episode_object_bounding_boxes.append([])
            self.episode_actions.append(action)
            with torch.no_grad():
                with torch.autocast(device_type=self.device, dtype=torch.float16):
                    masks = self.generator.generate(obs)
            self.episode_detected_masks.append([mask["segmentation"] for mask in masks])
            wandb.log({"data_collected": self.collected_data})
            for obj in self.env.objects:
                self.episode_object_types[-1].append(obj.category)
                self.episode_object_bounding_boxes[-1].append(obj.xywh)
            progress_bar.update(1)
            if terminated or truncated:
                self.store_episode()
                print(f"Finished {self.curr_episode_id - 1} episodes. ({self.collected_data})")
                obs, _ = self.env.reset()
        progress_bar.close()

    def store_episode(self) -> None:
        """
        Store the current episode to disk
        """
        file_name = f"{self.dataset_path}/{self.curr_episode_id}-{len(self.episode_frames)}.gz"
        np.savez_compressed(file_name,
                            episode_frames=np.array(self.episode_frames),
                            episode_object_types=np.array(self.episode_object_types),
                            episode_object_bounding_boxes=np.array(self.episode_object_bounding_boxes),
                            episode_detected_masks=np.array(self.episode_detected_masks),
                            episode_actions=np.array(self.episode_actions))
        self.curr_episode_id += 1
        episode_length = len(self.episode_frames)
        self.collected_data += episode_length
        wandb.log({"episode_length": episode_length})
        self.episode_frames = []
        self.episode_object_types = []
        self.episode_object_bounding_boxes = []
        self.episode_detected_masks = []
        self.episode_actions = []

    def determine_next_episode(self) -> None:
        """
        Determine the next episode id which hasnt been collected yet
        Sets curr_episode_id to the next episode id
        """
        for file in os.listdir(self.dataset_path):
            if file.endswith(".gz"):
                self.curr_episode_id = max(self.curr_episode_id, int(file.split("-")[0]) + 1)

    def get_collected_data(self) -> int:
        """
        Determine how much data we have already collected
        """
        data = 0
        for file in os.listdir(self.dataset_path):
            if file.endswith(".gz"):
                # all files have the format {id)-{length}.gz
                data += int(file.split("-")[1].split(".")[0])

        return data
