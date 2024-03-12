import os
from typing import Tuple
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

from src.data_collection.common import get_data_directory, get_id_from_episode_name, get_length_from_episode_name

class DataLoader:
    def __init__(self, game: str):
        self.dataset_path = get_data_directory(game)
        self.load_data()
        self.history_len = 4

    def load_data(self) -> None:
        """
        Load all the data from the disk
        """
        episodes_paths = [f for f in os.listdir(self.dataset_path) if f.endswith(".npz")]
        episode_counts = 0
        episode_lengths = []
        episode_data = {}
        for episode in episodes_paths:
            data = np.load(self.dataset_path + episode)
            episode_lengths.append(get_length_from_episode_name(episode))
            episode_id = get_id_from_episode_name(episode)
            assert episode_id not in episode_data, f"Episode {episode_id} already exists in the dataset"
            episode_data[episode_id] = (data["episode_frames"], data["episode_object_types"], data["episode_object_bounding_boxes"],
                                data["episode_detected_masks"], data["episode_actions"])
            episode_counts += 1

        self.episode_data = episode_data
        self.episode_weights = np.array(episode_lengths) / sum(episode_lengths)

    def sample(self, batch_size: int, time_steps: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        """
        Sample a batch of states from episodes
        Args:
        batch_size: Number of states to sample
        time_steps: Number of time steps to sample for bboxes
        Returns:
        Tuple containing stacked states [batch_size, stacked_frames=4, channels=3, H=128, W=128], Object bounding boxes [batch_size, num_objects, 4],
        Masks [batch_size, H=128, W=128], Actions [batch_size, 1]
        """
        episodes = np.random.choice(len(self.episode_data), size=batch_size, p=self.episode_weights)
        states = []
        object_bounding_boxes_list = []
        masks = []
        actions = []
        for episode in episodes:
            frames, _, object_bounding_boxes, detected_masks, episode_actions = self.episode_data[episode]
            start = np.random.randint(0, len(frames) - self.history_len)
            states.append(frames[start:start+self.history_len])
            object_bounding_boxes_list.append(object_bounding_boxes[start+self.history_len:start+self.history_len+time_steps])
            masks.append(detected_masks[start+self.history_len])
            actions.append(episode_actions[start+self.history_len])

        states = torch.from_numpy(np.array(states))
        states = states / 255
        states = states.permute(0, 4, 1, 2, 3)
        object_bounding_boxes_list = torch.from_numpy(np.array(object_bounding_boxes_list))
        object_bounding_boxes_list = object_bounding_boxes_list.squeeze(1).float()
        object_bounding_boxes_list /= torch.Tensor([states.shape[-2], states.shape[-1], states.shape[-2], states.shape[-1]]).float()

        states = states.reshape(*states.shape[:1], -1, *states.shape[3:])
        states = F.interpolate(states, (128, 128))
        states = states.reshape((-1, 12, 128, 128))
        masks = torch.from_numpy(np.array(masks))

        return states, object_bounding_boxes_list, masks, torch.from_numpy(np.array(actions))
