from typing import List, Tuple
import os
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from hydra.utils import to_absolute_path

from src.data_collection.common import get_data_directory, get_id_from_episode_name, get_length_from_episode_name


class DataLoader:
    def __init__(self, game: str, num_obj: int, history_len: int, train_pct: float = 0.7, val_pct: float = 0.15, test_pct: float = 0.15):
        assert train_pct + val_pct + test_pct == 1, "Train, validation and test percentages should sum to 1"
        self.dataset_path = get_data_directory(game)
        self.load_data()
        self.history_len = history_len 
        self.num_obj = num_obj
        self.num_train = int(((train_pct * len(self.frames))) // self.history_len) * self.history_len
        self.num_val = int(((val_pct * len(self.frames))) // self.history_len) * self.history_len
        self.num_test = len(self.frames) - self.num_train - self.num_val

    def load_data(self) -> None:
        """
        Load all the data from the disk
        """
        episodes_paths = [f for f in os.listdir(to_absolute_path(self.dataset_path)) if f.endswith(".npz")]
        self.episode_count = 0
        self.episode_lengths = []
        episode_ids = set()
        frames = []
        object_types = []
        object_bounding_boxes = []
        detected_masks = []
        actions = []
        last_idx = []
        for episode in episodes_paths:
            data = np.load(to_absolute_path(self.dataset_path + episode))
            self.episode_lengths.append(get_length_from_episode_name(episode))
            episode_id = get_id_from_episode_name(episode)
            assert episode_id not in episode_ids, f"Episode {episode_id} already exists in the dataset"
            episode_ids.add(episode_id)
            frames.append(data["episode_frames"])
            object_types.append(data["episode_object_types"])
            object_bounding_boxes.append(data["episode_object_bounding_boxes"])
            detected_masks.append(data["episode_detected_masks"])
            actions.append(data["episode_actions"])
            last_idx.append(data["episode_last_idx"])
            self.episode_count += 1
        self.frames = np.concatenate(frames, axis=0)
        self.object_types = np.concatenate(object_types, axis=0)
        self.object_bounding_boxes = np.concatenate(object_bounding_boxes, axis=0)
        self.detected_masks = np.concatenate(detected_masks, axis=0)
        self.actions = np.concatenate(actions, axis=0)
        self.last_idx = np.concatenate(last_idx, axis=0)

    def sample(self, batch_size: int, time_steps: int, device: str, data_type: str = "train") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch of states from episodes
        Args:
        batch_size: Number of states to sample
        time_steps: Number of time steps to sample for bboxes
        data_type: Type of data to sample from. Can be "train", "val" or "test"
        Returns:
        Tuple containing stacked states [batch_size, time_steps=4, channels=3, H=128, W=128], Object bounding boxes [batch_size, t, num_objects, 4],
        Masks [batch_size, num_obj, H=128, W=128], Actions [batch_size, t]
        - States are in the past
        - Masks are in the present
        - Bounding boxes, actions are in the future
        """
        if data_type == "train":
            start, end = 0, self.num_train
        elif data_type == "val":
            start, end = self.num_train, self.num_train + self.num_val
        elif data_type == "test":
            start, end = self.num_train + self.num_val, len(self.frames)
        frames = np.random.choice(np.arange(start, end), size=batch_size)
        states: List[npt.NDArray] = []
        object_bounding_boxes_list: List[npt.NDArray] = []
        masks: List[npt.NDArray] = []
        actions = []
        for idx in frames:
            idx = min(idx, len(self.frames) - time_steps)  # Ensure we don't go out of bounds
            idx = max(idx, self.history_len)
            frame = self.frames[idx - self.history_len : idx]
            object_bounding_boxes = self.object_bounding_boxes[idx : idx + time_steps]  # [T, O, 4] future bboxes
            mask = self.detected_masks[idx]  # [O, H, W]
            action = self.actions[idx : idx + time_steps]  # [T]
            last_idxs = self.last_idx[idx : idx + time_steps]  # [T, O]
            states.append(frame)
            objs = object_bounding_boxes[0].sum(-1) != 0  # [O]
            orderd_bbxs = np.zeros_like(object_bounding_boxes)  # [T, O, 4] ordered by the initial object they are tracking
            order = np.arange(objs.sum())  # [o]
            for t in range(time_steps):
                orderd_bbxs[t, order] = object_bounding_boxes[t, objs]
                order = last_idxs[t, order]
            object_bounding_boxes_list.append(orderd_bbxs)
            masks.append(mask)
            actions.append(action)

        states_tensor = torch.from_numpy(np.array(states)).to(device)
        states_tensor = states_tensor / 255
        states_tensor = states_tensor.permute(0, 1, 4, 2, 3)  # [B, T, C, H, W]
        object_bounding_boxes_tensor = torch.from_numpy(np.array(object_bounding_boxes_list)).to(device)
        object_bounding_boxes_tensor = object_bounding_boxes_tensor.float()
        w = states_tensor.shape[-2]
        h = states_tensor.shape[-1]
        object_bounding_boxes_tensor /= torch.Tensor([h, w, h, w]).to(device).float()
        object_bounding_boxes_tensor = object_bounding_boxes_tensor[:, :, : self.num_obj]

        states_tensor = states_tensor.reshape(*states_tensor.shape[:1], -1, *states_tensor.shape[3:])
        states_tensor = F.interpolate(states_tensor, (128, 128))
        states_tensor = states_tensor.reshape((-1, 3 * self.history_len, 128, 128))

        masks_tensor = torch.from_numpy(np.array(masks)).to(device)
        masks_tensor = F.one_hot(masks_tensor.long(), num_classes=self.num_obj + 1).float()[:, :, :, 1:]
        masks_tensor = masks_tensor.permute(0, 3, 1, 2)
        masks_tensor = F.interpolate(masks_tensor, (128, 128))

        return states_tensor, object_bounding_boxes_tensor, masks_tensor, torch.from_numpy(np.array(actions))
