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
                                data["episode_detected_masks"], data["episode_actions"])
            episode_counts += 1

        self.episode_data = episode_data
        self.episode_weights = np.array(episode_lengths) / sum(episode_lengths)

    def sample(self, batch_size: int, time_steps: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
            frames, _, object_bounding_boxes, detected_masks, episode_actions = self.episode_data[episode]
            start = np.random.randint(0, len(frames) - self.history_len - time_steps)
            base = start + self.history_len
            states.append(frames[start:base])
            obj_bbxs = object_bounding_boxes[base:base+time_steps]  # [T, O, 4]
            object_bounding_boxes_list.append(obj_bbxs)
            masks.append(detected_masks[base])
            actions.append(episode_actions[base:base+time_steps])

        states_tensor = torch.from_numpy(np.array(states))
        states_tensor  = states_tensor  / 255
        states_tensor  = states_tensor .permute(0, 4, 1, 2, 3)
        object_bounding_boxes_tensor= torch.from_numpy(np.array(object_bounding_boxes_list))
        object_bounding_boxes_tensor = object_bounding_boxes_tensor.float()
        w = states_tensor.shape[-2]
        h = states_tensor.shape[-1]
        object_bounding_boxes_tensor /= torch.Tensor([h, w, h, w]).float()
        object_bounding_boxes_tensor = object_bounding_boxes_tensor[:, :, :self.num_obj]

        states_tensor = states_tensor.reshape(*states_tensor.shape[:1], -1, *states_tensor.shape[3:])
        states_tensor  = F.interpolate(states_tensor , (128, 128))
        states_tensor  = states_tensor .reshape((-1, 12, 128, 128))

        masks_tensor = torch.from_numpy(np.array(masks))[:, :self.num_obj]

        return states_tensor, object_bounding_boxes_tensor, masks_tensor, torch.from_numpy(np.array(actions))
