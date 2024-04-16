from typing import List, Tuple
import os
import typing
import cv2
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
        self.episode_object_bounding_boxes: List[npt.NDArray] = []
        self.episode_object_xy: List[List[Tuple[int,int]]] = []
        self.episode_object_last_idx: List[List[int]] = []
        self.episode_detected_masks: List[npt.NDArray] = []  # uint8 list of (H, W) masks
        self.episode_actions: List[int] = []

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        weights = "FastSAM-s" if small_sam else "FastSAM-x"
        self.sam = FastSAM(f"./models/{weights}.pt")
        self.sam.to(self.device)
        self.model_name = weights

    def collect_data(self) -> None:
        """
        Collects data from the environment for a given number of steps.
        """
        progress_bar = tqdm(total=self.num_samples, desc="Collecting data")
        progress_bar.update(self.collected_data)
        obs, _ = self.env.reset()
        counter = 0
        orig_size = typing.cast(Tuple[int,int], obs.shape[:2])
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

            masks_idx, masks_center, masks_size, masks = self.get_masks(obs, padded_size, orig_size)

            # Ground truth
            np_object_bounding_boxes, object_types, object_xy, object_last_idx = self.get_ground_truth_data()

            self.episode_object_types.append(object_types)
            self.episode_object_bounding_boxes.append(np_object_bounding_boxes)
            self.episode_object_xy.append(object_xy)
            self.episode_object_last_idx.append(object_last_idx)
            # we must track the objects between frames
            pos_costs = (np_object_bounding_boxes[:, :2] + np_object_bounding_boxes[:, 2:] / 2)[:, np.newaxis, :] - masks_center[np.newaxis, :, :]
            pos_costs = np.linalg.norm(pos_costs, axis=2)
            size_costs = 10 * np.sqrt(np.abs(np_object_bounding_boxes[:, 2:].prod(axis=1)[:, np.newaxis] - masks_size[np.newaxis, :]))
            costs = pos_costs + size_costs
            num_objects = len(object_types)
            log_dir = {"data_collected": counter, "num_objects": num_objects}
            matched_masks = np.zeros((num_objects + 1, masks.shape[1], masks.shape[2]))
            while len(masks_idx) > 0 and num_objects > 0:
                min_pos = np.unravel_index(np.argmin(costs), costs.shape)
                matched_masks[min_pos[0] + 1] = masks[min_pos[1]]
                costs[min_pos[0]] = np.inf
                costs[:, min_pos[1]] = np.inf
                num_objects -= 1

            # uint8 array (H, W) of mask ids, assuming they do not overlap
            masks = matched_masks.argmax(axis=0).astype(np.uint8)
            self.episode_detected_masks.append(masks)

            wandb.log(log_dir)
            progress_bar.update(1)
            if terminated or truncated or counter == 10:
                self.store_episode()
                tqdm.write(f"Finished {self.curr_episode_id - 1} episodes. ({self.collected_data})")
                obs, _ = self.env.reset()
        progress_bar.close()

    def get_ground_truth_data(self) -> Tuple[npt.NDArray, List[str], List[Tuple[int,int]], List[int]]:
        """
        Extract the ground truth OCAtari Information from the environment
        """
        object_types = []
        object_bounding_boxes = []
        object_xy = []
        object_last_idx = []
        for obj in self.env.objects:
            x, y, w, h = obj.xywh
            if (x == 0 and y == 0) or (w < 0) or (h < 0):
                continue
            object_types.append(obj.category)
            object_bounding_boxes.append(np.array(obj.xywh))
            object_xy.append(obj.xy)
            last_idx = -1 if len(self.episode_object_xy) == 0 or obj.prev_xy == (0, 0) \
                or obj.prev_xy not in self.episode_object_xy[-1] \
                else self.episode_object_xy[-1].index(obj.prev_xy)
            object_last_idx.append(last_idx)
        np_object_bounding_boxes = np.array(object_bounding_boxes)
        return np_object_bounding_boxes, object_types, object_xy, object_last_idx

    def get_masks(self, obs: np.ndarray, padded_size: Tuple[int,int], orig_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calls SAM to get the masks from the image
        Returns:
        Tuple of masks_idx, masks_center, masks_size, masks
        """
        # SAM masks
        with torch.no_grad():# , torch.autocast(device_type=self.device, dtype=torch.float16):
            # Upscale SAM input
            upscaled = cv2.resize(obs, (1024, 1024))
            results = self.sam(upscaled, retina_masks=True, imgsz=upscaled.shape[0], conf=0.4, iou=0.9, verbose=False)
        if results is None:
            masks = np.zeros((1, padded_size[0], padded_size[1]), dtype=bool)
        else:
            masks_t = results[0].masks.data.unsqueeze(0).to(torch.float32)  # (N, H, W)
            masks_t = F.interpolate(masks_t, padded_size, mode="nearest").to(bool).squeeze(0)
            masks = self.filter_and_sort_masks(orig_size, masks_t).cpu().numpy()
            num_masks = (masks.sum(-1).sum(-1) != 0).sum()
            masks_idx = np.arange(num_masks)  # this is used later in the matching
            masks_center = np.zeros((num_masks, 2))
            masks_size = np.zeros(num_masks)
            for i in range(num_masks):
                mask = masks[i]
                y, x = np.where(mask)
                masks_center[i, 0] = x.mean()
                masks_center[i, 1] = y.mean()
                masks_size[i] = len(x)
        return masks_idx, masks_center, masks_size, masks

    def filter_and_sort_masks(self, orig_size: Tuple[int, int], masks: torch.Tensor) -> torch.Tensor:
        """
        Filter masks for duplicates and sort them by size.
        Args:
            orig_size: (H, W) tuple, the original size of the image
            masks: (N, H, W) binary masks
        Returns:
            List[npt.NDArray] of HxW binary masks
        """
        good_masks = torch.zeros(
            size=(masks.shape[0], orig_size[0], orig_size[1]),
            dtype=torch.bool,
            device=self.device
        )  # (num_objects, H, W)
        indiv_masks = sorted(
            [masks[i,:orig_size[0],:orig_size[1]] for i in range(masks.shape[0]) if masks[i].sum() > 0 and masks[i].sum() < orig_size[0] * orig_size[1] / 2],
            key=lambda m: m.sum().item(),
            reverse=True,
        )
        c = 0
        for mask in indiv_masks:
            # ensure we dont have duplicate masks
            if c == 0 or max((torch.bitwise_and(mask, m)).sum().item() for m in good_masks[:c]) / mask.sum() < 0.8:
                good_masks[c] = mask
                c += 1
        return good_masks

    def store_episode(self) -> None:
        """
        Store the current episode to disk
        """
        file_name = f"{self.dataset_path}/{self.curr_episode_id}-{len(self.episode_frames)}-{self.model_name}"
        episode_object_types = np.array(
            [np.pad(objs_types, (0, self.max_num_objects - len(objs_types)), constant_values="") for objs_types in self.episode_object_types]
        )
        episode_object_bounding_boxes = np.array(
            [np.pad(objs_bb, ((0, self.max_num_objects - len(objs_bb)), (0, 0))) for objs_bb in self.episode_object_bounding_boxes]
        )
        episode_object_last_idx = np.array(
            [np.pad(objs_last_idx, ((0, self.max_num_objects - len(objs_last_idx))), constant_values=-1) for objs_last_idx in self.episode_object_last_idx]
        )

        np.savez_compressed(
            file_name,
            episode_frames=np.array(self.episode_frames),
            episode_object_types=episode_object_types,
            episode_object_bounding_boxes=episode_object_bounding_boxes,
            episode_detected_masks=np.array(self.episode_detected_masks),
            episode_actions=np.array(self.episode_actions),
            episode_last_idx=episode_object_last_idx,
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
        self.episode_object_xy = []
        self.episode_object_last_idx = []

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
