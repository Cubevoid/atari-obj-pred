from typing import Any
import tkinter
import customtkinter as ctk  # type: ignore
import cv2  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.data_collection.data_loader import DataLoader

# generate a list of 32 distinct colors for matplotlib
color_map = [plt.cm.tab20(i) for i in np.linspace(0, 1, 33)]  # type: ignore[attr-defined] # pylint: disable=no-member


class Visualizer:
    def __init__(self, game: str) -> None:
        self.data_loader = DataLoader(game, 32)

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
        self.root.mainloop()

    def set_display_mode(self) -> None:
        self.update_surface(None)

    def update_data_slider(self, _: Any) -> None:
        episode = self.data_loader.episode_data[int(self.episode_slider.get()) - 1][0]
        self.data_slider.configure(number_of_steps=len(episode)-1, to=len(episode))

    def update_surface(self, _: Any) -> None:
        episode_idx = int(self.data_slider.get()) - 1
        frame, types, boxes, masks, actions = self.data_loader.episode_data[int(self.episode_slider.get()) - 1]
        frame, types, boxes, masks, actions = frame[episode_idx], types[episode_idx], boxes[episode_idx], masks[episode_idx], actions[episode_idx]
        frame = frame.astype(np.float32) / 255.
        mode = self.radio_var.get()
        if mode in [2, 3, 5]:
            if mode in [2, 5]:
                frame = np.zeros_like(frame)
            masks = F.one_hot(torch.Tensor(masks).long(), 33).float()[:, :, 1:].bool().numpy()
            for i in range(masks.shape[-1]):
                mask = masks[:, :, i]
                img = np.zeros_like(frame)
                img[mask] = color_map[i][:3]
                frame += img * 0.5
        frame = frame.clip(0, 1)
        if mode in [4, 5]:
            for i, box in enumerate(boxes):
                x, y, w, h = box
                if x != 0 or y != 0:
                    frame = cv2.rectangle(frame, (x, y), (x+w, y+h), color_map[i], 1)  # pylint: disable=no-member
        self.ax.imshow(frame)
        self.ax.axis("off")
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        self.canvas.draw()
        self.root.update()
