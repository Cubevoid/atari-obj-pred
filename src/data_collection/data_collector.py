from typing import List, Tuple
import os
import numpy as np
import numpy.typing as npt
from ocatari.core import OCAtari
from ocatari.utils import load_agent
from fastsam import FastSAM  # type: ignore
import torch
from torch.nn import functional as F
from tqdm import tqdm
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
            self.env = OCAtari(game, mode="revised", hud=True, obs_mode="dqn")
            self.agent = load_agent(f"./models/dqn_{game}.gz", self.env.action_space.n)  # type: ignore
        self.num_samples = num_samples
        self.max_num_objects = max_num_objects

        self.dataset_path = get_data_directory(game)
        os.makedirs(self.dataset_path, exist_ok=True)

        self.curr_episode_id = 0
        self.determine_next_episode()
        self.collected_data = self.get_collected_data()
        self.episode_frames: List[npt.NDArray] = []
        self.episode_object_types: List[List[str]] = []
        self.episode_object_bounding_boxes: List[List[npt.NDArray]] = []
        self.episode_detected_masks: List[npt.NDArray] = []  # uint8 list of (H, W) masks
        self.episode_actions: List[int] = []

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        weights = "FastSAM-s.pt" if small_sam else "FastSAM-x.pt"
        self.sam = FastSAM(f"./models/{weights}")
        self.sam.to(self.device)

    def resize_masks(self, masks: torch.Tensor, size: Tuple[int, ...]) -> torch.Tensor:
        """
        Resize the masks to 128x128
        Args:
            masks: (C, H, W) tensor
        Returns:
            (B, C, 128, 128) tensor
        """
        return F.interpolate(masks.unsqueeze(1), size=size, mode="nearest").squeeze(1)

    def collect_data(self) -> None:
        """
        Collects data from the environment for a given number of steps.
        """
        progress_bar = tqdm(total=self.num_samples, desc="Collecting data")
        progress_bar.update(self.collected_data)
        obs, _ = self.env.reset()
        counter = 0
        orig_size = obs.shape[:2]
        while self.collected_data < self.num_samples:
            counter += 1
            # Generate game frame
            action = self.agent.draw_action(self.env.dqn_obs) if self.agent else self.env.action_space.sample()  # type: ignore
            obs, _, terminated, truncated, _ = self.env.step(action)
            self.episode_frames.append(obs)
            self.episode_actions.append(action)
            # Pad the image to be a multiple of 32
            padded_size = (orig_size[0] + 31) // 32 * 32, (orig_size[1] + 31) // 32 * 32
            obs = np.pad(obs, ((0, padded_size[0] - orig_size[0]), (0, padded_size[1] - orig_size[1]), (0, 0)))
            # SAM masks
            with torch.no_grad(), torch.autocast(device_type=self.device, dtype=torch.float16):
                results = self.sam(obs, retina_masks=True, imgsz=max(padded_size), conf=0.4, iou=0.9, verbose=False)
            masks = results[0].masks.data.cpu().numpy().astype(bool)  # (N, H, W)
            masks = self.filter_and_sort_masks(orig_size, masks)
            masks = np.pad(masks, ((1, 0), (0, 0), (0, 0)))  # add background "mask"
            # uint8 array (H, W) of mask ids, assuming they do not overlap
            masks = masks.argmax(axis=0).astype(np.uint8)
            self.episode_detected_masks.append(masks)

            wandb.log({"data_collected": counter})

            # Ground truth
            self.episode_object_types.append([])
            self.episode_object_bounding_boxes.append([])
            for obj in self.env.objects:
                self.episode_object_types[-1].append(obj.category)
                self.episode_object_bounding_boxes[-1].append(np.array(obj.xywh))

            progress_bar.update(1)
            if terminated or truncated:
                self.store_episode()
                tqdm.write(f"Finished {self.curr_episode_id - 1} episodes. ({self.collected_data})")
                obs, _ = self.env.reset()
        progress_bar.close()

    def filter_and_sort_masks(self, orig_size: Tuple[int, ...], masks: npt.NDArray) -> npt.NDArray:
        """
        Filter masks for duplicates and sort them by size.
        Args:
            orig_size: (H, W) tuple, the original size of the image
            masks: (N, H, W) binary masks
        Returns:
            List[npt.NDArray] of HxW binary masks
        """
        padded_masks = np.zeros(
            (masks.shape[0], orig_size[0], orig_size[1]),
            dtype=bool,
        )  # (num_objects, H, W)
        indiv_masks = sorted(
            [masks[i,:orig_size[0],:orig_size[1]] for i in range(masks.shape[0]) if masks[i].sum() > 0 and masks[i].sum() < orig_size[0] * orig_size[1] / 2],
            key=lambda m: m.sum(),
            reverse=True,
        )
        c = 0
        for mask in indiv_masks:
            # ensure we dont have duplicate masks
            if c == 0 or max((np.bitwise_and(mask, m)).sum() for m in padded_masks[:c]) / mask.sum() < 0.8:
                padded_masks[c] = mask
                c += 1
        return padded_masks

    def store_episode(self) -> None:
        """
        Store the current episode to disk
        """
        file_name = f"{self.dataset_path}/{self.curr_episode_id}-{len(self.episode_frames)}"
        episode_object_types = np.array(
            [np.pad(objs_types, (0, self.max_num_objects - len(objs_types)), constant_values="") for objs_types in self.episode_object_types]
        )
        episode_object_bounding_boxes = np.array(
            [np.pad(objs_bb, ((0, self.max_num_objects - len(objs_bb)), (0, 0))) for objs_bb in self.episode_object_bounding_boxes]
        )
        np.savez_compressed(
            file_name,
            episode_frames=np.array(self.episode_frames),
            episode_object_types=episode_object_types,
            episode_object_bounding_boxes=episode_object_bounding_boxes,
            episode_detected_masks=self.episode_detected_masks,
            episode_actions=np.array(self.episode_actions),
        )
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
        return count
