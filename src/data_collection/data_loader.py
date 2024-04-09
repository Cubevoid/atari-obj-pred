from typing import List, Tuple
import os
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from hydra.utils import to_absolute_path

from src.data_collection.common import get_data_directory, get_id_from_episode_name, get_length_from_episode_name


class DataLoader:
    def __init__(self, game: str, num_obj: int):
        self.dataset_path = get_data_directory(game)
        self.load_data()
        self.history_len = 4
        self.num_obj = num_obj

    def load_data(self) -> None:
        """
        Load all the data from the disk
        """
        episodes_paths = [f for f in os.listdir(to_absolute_path(self.dataset_path)) if f.endswith(".npz")]
        episode_counts = 0
        episode_lengths = []
        episode_data = {}
        for episode in episodes_paths:
            data = np.load(to_absolute_path(self.dataset_path + episode))
            episode_lengths.append(get_length_from_episode_name(episode))
            episode_id = get_id_from_episode_name(episode)
            assert episode_id not in episode_data, f"Episode {episode_id} already exists in the dataset"
            episode_data[episode_id] = (data["episode_frames"], data["episode_object_types"], data["episode_object_bounding_boxes"],
                                data["episode_detected_masks"], data["episode_actions"], data["episode_last_idx"])
            episode_counts += 1

        self.episode_data = episode_data
        self.episode_weights = np.array(episode_lengths) / sum(episode_lengths)

    def sample(self, batch_size: int, time_steps: int, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch of states from episodes
        Args:
        batch_size: Number of states to sample
        time_steps: Number of time steps to sample for bboxes
        Returns:
        Tuple containing stacked states [batch_size, stacked_frames=4, channels=3, H=128, W=128], Object bounding boxes [batch_size, t, num_objects, 4],
        Masks [batch_size, num_obj, H=128, W=128], Actions [batch_size, t]
        """
        episodes = np.random.choice(len(self.episode_data), size=batch_size, p=self.episode_weights)
        states: List[npt.NDArray] = []
        object_bounding_boxes_list: List[npt.NDArray] = []
        masks: List[npt.NDArray] = []
        actions = []
        for episode in episodes:
            stacked_state, orderd_bbxs, cur_mask, action = self.get_step(episode, time_steps)

            states.append(stacked_state)
            object_bounding_boxes_list.append(orderd_bbxs)
            masks.append(cur_mask)
            actions.append(action)

        states_tensor = torch.from_numpy(np.array(states)).to(device)
        states_tensor = states_tensor / 255
        states_tensor = states_tensor.permute(0, 4, 1, 2, 3)
        object_bounding_boxes_tensor = torch.from_numpy(np.array(object_bounding_boxes_list)).to(device)
        object_bounding_boxes_tensor = object_bounding_boxes_tensor.float()
        w = states_tensor.shape[-2]
        h = states_tensor.shape[-1]
        object_bounding_boxes_tensor /= torch.Tensor([h, w, h, w]).to(device).float()
        object_bounding_boxes_tensor = object_bounding_boxes_tensor[:, :, :self.num_obj]

        states_tensor = states_tensor.reshape(*states_tensor.shape[:1], -1, *states_tensor.shape[3:])
        states_tensor = F.interpolate(states_tensor, (128, 128))
        states_tensor = states_tensor.reshape((-1, 12, 128, 128))

        masks_tensor = torch.from_numpy(np.array(masks)).to(device)
        masks_tensor = F.one_hot(masks_tensor.long(), num_classes=self.num_obj + 1).float()[:, :, :, 1:]
        masks_tensor = masks_tensor.permute(0, 3, 1, 2)
        masks_tensor = F.interpolate(masks_tensor, (128, 128))

        return states_tensor, object_bounding_boxes_tensor, masks_tensor, torch.from_numpy(np.array(actions))


    def get_step(self, episode: int, time_steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the frame from the episode
        Args:
        episode: Episode number
        Returns:
        Frame as numpy array, bounding boxes, current FastSAM masks, Action
        """
        frames, _, object_bounding_boxes, detected_masks, episode_actions, last_idxs = self.episode_data[episode]
        start = np.random.randint(0, len(frames) - self.history_len - time_steps)
        base = start + self.history_len
        obj_bbxs = object_bounding_boxes[base:base + time_steps]  # [T, O, 4]
        objs = obj_bbxs[0].sum(-1) != 0  # [O]
        orderd_bbxs = np.zeros_like(obj_bbxs)  # [T, O, 4] ordered by the initial object they are tracking
        order = np.arange(objs.sum())  # [o]
        for t in range(time_steps):
            orderd_bbxs[t, order] = obj_bbxs[t, objs]
            order = last_idxs[base + t, order]
        stacked_state = frames[start:base]
        cur_mask = detected_masks[base]
        action = episode_actions[base:base + time_steps]
        return stacked_state, orderd_bbxs, cur_mask, action
