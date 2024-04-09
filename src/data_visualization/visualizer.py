from typing import Any, List, Tuple
import tkinter
from colorsys import hls_to_rgb
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
import customtkinter as ctk  # type: ignore
import cv2  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.data_collection.data_loader import DataLoader
from src.model.feat_extractor import FeatureExtractor
from src.model.predictor import Predictor

# generate a list of 32 distinct colors for matplotlib
def get_distinct_colors(n: int) -> List[Tuple[float, float, float]]:
    colors = []

    for i in np.arange(0., 360., 360. / n):
        h = i / 360.
        l = (50 + np.random.rand() * 10) / 100.
        s = (90 + np.random.rand() * 10) / 100.
        colors.append(hls_to_rgb(h, l, s))

    return colors
# color_map = [plt.cm.Set1(i) for i in np.linspace(0, 1, 33)]  # type: ignore[attr-defined] # pylint: disable=no-member
color_map = get_distinct_colors(8)


class Visualizer:
    def __init__(self, cfg: DictConfig) -> None:
        self.data_loader = DataLoader(cfg.game, cfg.num_objects)

        feature_extractor_state = torch.load("models/trained/1711831906_feat_extract.pth")
        self.feature_extractor = FeatureExtractor(num_objects=cfg.num_objects)
        self.feature_extractor.load_state_dict(feature_extractor_state)
        predictor_state = torch.load("models/trained/1711831906_transformer_predictor.pth")
        self.predictor = Predictor()
        self.predictor.load_state_dict(predictor_state)

        ctk.set_appearance_mode("dark")
        self.root = ctk.CTk()
        self.root.geometry("1200x800+200x200")
        self.root.title("Data Visualizer")
        self.root.update()
        self.frame = ctk.CTkFrame(master=self.root,
                                  height= self.root.winfo_height()*0.66,
                                  width = self.root.winfo_width()*0.66,
                                  fg_color="darkblue")
        self.frame.place(relx=0.33, rely=0.025)
        num_episodes = len(self.data_loader.episode_data)
        self.episode_slider = ctk.CTkSlider(master=self.root,
                                    width=300,
                                    height=20,
                                    from_=1,
                                    to=num_episodes,
                                    number_of_steps=num_episodes-1,
                                    command=self.update_surface)
        self.episode_slider.place(relx= 0.025,rely=0.5)
        self.data_slider = ctk.CTkSlider(master=self.root,
                                    width=300,
                                    height=20,
                                    from_=1,
                                    to=num_episodes,
                                    number_of_steps=num_episodes-1,
                                    command=self.update_surface)

        self.radio_var = tkinter.IntVar(value=1)
        radiobutton_1 = ctk.CTkRadioButton(self.root, text="Image",
                                                    command=self.set_display_mode, variable= self.radio_var, value=1)
        radiobutton_1.place(relx= 0.025,rely=0.1)
        radiobutton_2 = ctk.CTkRadioButton(self.root, text="SAM Masks",
                                             command=self.set_display_mode, variable= self.radio_var, value=2)
        radiobutton_2.place(relx= 0.025,rely=0.15)
        radiobutton_3 = ctk.CTkRadioButton(self.root, text="SAM Masks + Image",
                                             command=self.set_display_mode, variable= self.radio_var, value=3)
        radiobutton_3.place(relx= 0.025,rely=0.2)
        radiobutton_4 = ctk.CTkRadioButton(self.root, text="Groundtruth",
                                             command=self.set_display_mode, variable= self.radio_var, value=4)
        radiobutton_4.place(relx= 0.025,rely=0.25)
        radiobutton_5 = ctk.CTkRadioButton(self.root, text="SAM Mask + Groundtruth",
                                             command=self.set_display_mode, variable= self.radio_var, value=5)
        radiobutton_5.place(relx= 0.025,rely=0.3)

        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(6,6)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().place(relx=0.33, rely=0.025)

        self.data_slider.place(relx= 0.025,rely=0.75)
        self.update_data_slider(None)
        # add event handler to episode slider to update max number of data slider
        self.episode_slider.bind("<ButtonRelease-1>", self.update_data_slider)
        self.update_surface(None)
        self.root.mainloop()

    def set_display_mode(self) -> None:
        self.update_surface(None)

    def update_data_slider(self, _: Any) -> None:
        episode = self.data_loader.episode_data[int(self.episode_slider.get()) - 1][0]
        self.data_slider.configure(number_of_steps=len(episode)-1, to=len(episode))

    def update_surface(self, _: Any) -> None:
        episode_idx = int(self.data_slider.get()) - 1
        frame, types, boxes, masks, actions, _ = self.data_loader.episode_data[int(self.episode_slider.get()) - 1]  # type: ignore
        frame, types, boxes, masks, actions = frame[episode_idx], types[episode_idx], boxes[episode_idx], masks[episode_idx], actions[episode_idx]
        masks = F.one_hot(torch.from_numpy(masks).long()).movedim(-1,0).numpy()[1:]  # the 0 mask is the background [O, W, H]
        frame = frame.astype(np.float32) / 255.
        orig_img = np.array(frame)
        mode = self.radio_var.get()
        if mode in [2, 3, 5]:
            if mode in [2, 5]:
                frame = np.zeros_like(frame)
            for i, mask in enumerate(masks):
                img = np.zeros_like(frame)
                img[mask == 1] = color_map[i][:3]
                frame += img * 0.5
                x, y, _, _ = boxes[i]
                if mode == 5:
                    mask_ys, mask_xs = np.nonzero(mask == 1)
                    if mask_xs.size > 0:
                        frame = cv2.arrowedLine(frame, (int(mask_xs.mean()), int(mask_ys.mean())), (x,y), color_map[i], 1)  # pylint: disable=no-member
        frame = frame.clip(0, 1)
        if mode in [4, 5]:
            for i, box in enumerate(boxes):
                x, y, w, h = box
                if x != 0 or y != 0:
                    frame = cv2.rectangle(frame, (x, y), (x+w, y+h), color_map[i], 1)  # pylint: disable=no-member

        # visualize predictions
        if self.predictor is not None:
            frame = frame * 0.5
            with torch.no_grad():
                frame_tensor = torch.from_numpy(orig_img).permute(2, 0, 1).unsqueeze(0).float()
                masks_tensor = torch.from_numpy(masks).unsqueeze(0).float()
                features = self.feature_extractor(frame_tensor, masks_tensor)
                predictions = self.predictor(features)
                for i, prediction in enumerate(predictions[0]):
                    x, y = prediction
                    if x != 0 or y != 0:
                        frame = cv2.circle(frame, (int(x), int(y)), 2, color_map[i], 2)

        self.ax.imshow(frame)
        self.ax.axis("off")
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        self.canvas.draw()
        self.root.update()
