from typing import List
import os
import cv2
import numpy as np
import numpy.typing as npt
from ocatari.core import OCAtari
from ocatari.utils import load_agent
from fastsam import FastSAM, FastSAMPrompt
import torch
from torch.nn import functional as F
import tqdm
import wandb

from src.data_collection.gen_simple_test_data import SimpleTestData
from src.data_collection.common import get_data_directory, get_length_from_episode_name

class DataCollector:
    def __init__(self, game: str, num_samples: int, max_num_objects: int, small_sam: bool = False) -> None:
        self.game = game
        if game == "SimpleTestData":
            self.env = SimpleTestData()
            self.agent = None
        else:
            self.env = OCAtari(game, mode="revised", hud=True, obs_mode='dqn')
            self.agent = load_agent(f"./models/dqn_{game}.gz", self.env.action_space.n)  # type: ignore
        self.num_samples = num_samples
        self.max_num_objects = max_num_objects

        self.dataset_path = get_data_directory(game)
        os.makedirs(self.dataset_path, exist_ok=True)

        self.curr_episode_id = 0
        self.determine_next_episode()
        self.collected_data = self.get_collected_data()
        self.episode_frames: List[npt.NDArray] = []
        self.episode_object_types : List[List[str]] = []
        self.episode_object_bounding_boxes : List[List[npt.NDArray]]= []
        self.episode_detected_masks : List[List[npt.NDArray]] = []
        self.episode_actions : List[int] = []

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        weights = "FastSAM-s.pt" if small_sam else "FastSAM-x.pt"
        self.sam = FastSAM(f"./models/{weights}")
        self.sam.to(self.device)

    def resize_masks(self, masks: npt.NDArray, size=(128, 128)) -> torch.tensor:
        """
        Resize the masks to 128x128
        Args:
            masks: (C, H, W) tensor
        Returns:
            (B, C, 128, 128) tensor
        """
        return F.interpolate(masks.unsqueeze(1), size=size, mode='nearest').squeeze(1)

    def collect_data(self) -> None:
        """
        Collects data from the environment for a given number of steps.
        """
        progress_bar = tqdm.tqdm(total=self.num_samples)
        progress_bar.update(self.collected_data)
        obs, _ = self.env.reset()
        counter = 0
        while self.collected_data < self.num_samples:
            counter += 1
            action = self.agent.draw_action(self.env.dqn_obs) if self.agent else self.env.action_space.sample()  # type: ignore
            obs, _, terminated, truncated, _ = self.env.step(action)
            self.episode_frames.append(obs)
            self.episode_object_types.append([])
            self.episode_object_bounding_boxes.append([])
            self.episode_actions.append(action)
            orig_size = obs.shape[:2]
            with torch.no_grad():
                with torch.autocast(device_type=self.device, dtype=torch.float16):
                    results = self.sam(obs, retina_masks=True, imgsz=max(orig_size), conf=0.4, iou=0.9, verbose=False)
                    masks = [
                        self.resize_masks(res.masks.data, orig_size).bool().cpu().numpy() for res in results
                    ]  # a B-len list of [N, H, W]
            self.episode_detected_masks.extend(masks)
            wandb.log({"data_collected": counter})
            for obj in self.env.objects:
                self.episode_object_types[-1].append(obj.category)
                self.episode_object_bounding_boxes[-1].append(np.array(obj.xywh))
            progress_bar.update(1)
            if terminated or truncated:
                self.store_episode()
                print(f"Finished {self.curr_episode_id - 1} episodes. ({self.collected_data})")
                obs, _ = self.env.reset()
        progress_bar.close()

        # filter and sort
        padded_masks = np.zeros((self.collected_data, self.max_num_objects, orig_size[0], orig_size[1]), dtype=bool)
        for i, img_masks in enumerate(masks):
            indiv_masks = sorted([mask for mask in img_masks if mask.sum() > 0 and mask.sum() < orig_size[0] * orig_size[1] / 2], key=lambda m: m.sum(), reverse=True)[:self.num_slots]
            c = 0
            for mask in indiv_masks:
                if c == 0 or max([(np.bitwise_and(mask, m)).sum() for m in padded_masks[i,:c]]) / mask.sum() < 0.8:  # ensure we dont have duplicate masks
                    padded_masks[i, c] = mask
                    c += 1
        self.episode_detected_masks = padded_masks

    def store_episode(self) -> None:
        """
        Store the current episode to disk
        """
        file_name = f"{self.dataset_path}/{self.curr_episode_id}-{len(self.episode_frames)}"
        episode_object_types = np.array([np.pad(objs_types, (0, self.max_num_objects - len(objs_types)), constant_values="")
                                         for objs_types in self.episode_object_types])
        episode_object_bounding_boxes = np.array([np.pad(objs_bb, ((0, self.max_num_objects - len(objs_bb)), (0,0)), constant_values=0)
                                                  for objs_bb in self.episode_object_bounding_boxes])
        episode_detected_masks = np.array([np.pad(masks, ((0, self.max_num_objects - len(masks)), (0,0), (0,0)), constant_values=0)
                                           for masks in self.episode_detected_masks])
        np.savez_compressed(file_name,
                            episode_frames=np.array(self.episode_frames),
                            episode_object_types=episode_object_types,
                            episode_object_bounding_boxes=episode_object_bounding_boxes,
                            episode_detected_masks=episode_detected_masks,
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
            if file.endswith(".npz"):
                self.curr_episode_id = max(self.curr_episode_id, int(file.split("-")[0]) + 1)

    def get_collected_data(self) -> int:
        """
        Determine how much data we have already collected
        """
        count = 0
        for file in os.listdir(self.dataset_path):
            if file.endswith(".npz"):
                count += get_length_from_episode_name(file)
        return 0
        return count
