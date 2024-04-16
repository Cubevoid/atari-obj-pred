from typing import Any, List, Tuple
import tkinter
from colorsys import hls_to_rgb
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
import customtkinter as ctk  # type: ignore
import cv2  # type: ignore
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from hydra.utils import instantiate

from src.data_collection.data_loader import DataLoader


# generate a list of 32 distinct colors for matplotlib
def get_distinct_colors(n: int) -> List[Tuple[float, float, float]]:
    colors = []
    for i in np.arange(0.0, 360.0, 360.0 / n):
        h = i / 360.0
        l = (50 + np.random.rand() * 10) / 100.0
        s = (90 + np.random.rand() * 10) / 100.0
        colors.append(hls_to_rgb(h, l, s))
    return colors

color_map = get_distinct_colors(32)

class Visualizer:
    def __init__(self, cfg: DictConfig) -> None:
        self.data_loader = DataLoader(cfg.game, cfg.num_objects, cfg.data_loader.history_len, max_data=10000)
        self.time_steps = cfg.time_steps
        self.history_len = cfg.data_loader.history_len
        # t = 1712855108  # pong - residual predictor
        # t = 1712959859  # pong - predictor
        # t = 1712962140  # pong - residual predictor t=20
        # t = 1712963786  # pong - residual predictor, gt_pos
        # t = 1712969070  # pong - residual, 8 history
        # t = 1712969847  # pong - residual, 20k
        t = 1712970451  # pong - residual, 20k, t=20

        try:
            feature_extractor_state = torch.load(f"models/trained/{cfg.game}/{t}_feat_extract.pth", map_location='cpu')
            self.feature_extractor = instantiate(cfg.feature_extractor, num_objects=cfg.num_objects, history_len=cfg.data_loader.history_len)
            self.feature_extractor.load_state_dict(feature_extractor_state)
            self.predictor = instantiate(cfg.predictor, time_steps=cfg.time_steps, log=False)
            predictor_state = torch.load(f"models/trained/{cfg.game}/{t}_{type(self.predictor).__name__}.pth", map_location='cpu')
            self.predictor.load_state_dict(predictor_state)
        except FileNotFoundError as e:
            print(e)
            self.predictor = None
        ctk.set_appearance_mode("dark")
        self.root = ctk.CTk()
        self.root.geometry("1200x800+200x200")
        self.root.title("Data Visualizer")
        self.root.update()
        self.frame = ctk.CTkFrame(master=self.root, height=self.root.winfo_height() * 0.66, width=self.root.winfo_width() * 0.66, fg_color="darkblue")
        self.frame.place(relx=0.33, rely=0.025)
        self.data_slider = ctk.CTkSlider(
            master=self.root,
            width=300,
            height=20,
            from_=1,
            to=len(self.data_loader.frames),
            number_of_steps=len(self.data_loader.frames) - 1,
            command=self.update_surface,
        )

        self.display_type = tkinter.IntVar(value=1)
        radiobutton_1 = ctk.CTkRadioButton(self.root, text="Image", command=self.set_display_mode, variable=self.display_type, value=1)
        radiobutton_1.place(relx=0.025, rely=0.1)
        radiobutton_2 = ctk.CTkRadioButton(self.root, text="SAM Masks", command=self.set_display_mode, variable=self.display_type, value=2)
        radiobutton_2.place(relx=0.025, rely=0.15)
        radiobutton_3 = ctk.CTkRadioButton(self.root, text="SAM Masks + Image", command=self.set_display_mode, variable=self.display_type, value=3)
        radiobutton_3.place(relx=0.025, rely=0.2)
        radiobutton_4 = ctk.CTkRadioButton(self.root, text="Groundtruth", command=self.set_display_mode, variable=self.display_type, value=4)
        radiobutton_4.place(relx=0.025, rely=0.25)
        radiobutton_5 = ctk.CTkRadioButton(self.root, text="SAM Mask + Groundtruth", command=self.set_display_mode, variable=self.display_type, value=5)
        radiobutton_5.place(relx=0.025, rely=0.3)

        self.show_prediction = tkinter.IntVar(value=0)
        radiobutton_6 = ctk.CTkRadioButton(self.root, text="No Prediction", command=self.set_display_mode, variable=self.show_prediction, value=0)
        radiobutton_6.place(relx=0.025, rely=0.4)
        radiobutton_7 = ctk.CTkRadioButton(self.root, text="Show Prediction", command=self.set_display_mode, variable=self.show_prediction, value=1)
        radiobutton_7.place(relx=0.025, rely=0.45)
        radiobutton_8 = ctk.CTkRadioButton(self.root, text="Show Prediction + GT", command=self.set_display_mode, variable=self.show_prediction, value=2)
        radiobutton_8.place(relx=0.025, rely=0.5)

        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(6, 6)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().place(relx=0.33, rely=0.025)

        self.data_slider.place(relx=0.025, rely=0.75)
        self.update_surface(None)
        self.root.mainloop()

    def set_display_mode(self) -> None:
        self.update_surface(None)

    def update_surface(self, _: Any) -> None:
        frame_idx = int(self.data_slider.get()) - 1
        frame = self.data_loader.frames[frame_idx]
        boxes = self.data_loader.object_bounding_boxes[frame_idx]
        masks = self.data_loader.detected_masks[frame_idx]
        masks = F.one_hot(torch.from_numpy(masks).long()).movedim(-1, 0).numpy()[1:]  # the 0 mask is the background [O, W, H]
        frame = frame.astype(np.float32) / 255.0
        mode = self.display_type.get()
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
                            frame = cv2.arrowedLine(frame, (int(mask_xs.mean()), int(mask_ys.mean())), (x, y), color_map[i], 1)  # pylint: disable=no-member
        frame = frame.clip(0, 1)
        if mode in [4, 5]:
            for i, box in enumerate(boxes):
                x, y, w, h = box
                if x != 0 or y != 0:
                    frame = cv2.rectangle(frame, (x, y), (x+w, y+h), color_map[i], 1)  # pylint: disable=no-member
        if self.predictor is not None and self.show_prediction.get() != 0:
            frame = self.visualize_prediction(frame, frame_idx)
        self.ax.imshow(frame)
        self.ax.axis("off")
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        self.canvas.draw()
        self.root.update()

    def visualize_prediction(self, frame: npt.NDArray, frame_idx: int) -> npt.NDArray:
        frame = frame * 0.5
        m_frame, m_bbxs, m_masks, _= self.data_loader.sample_idxes(self.time_steps, "cpu", [frame_idx])
        positions = m_bbxs[:, :, :, :2]  # [B, H + T, O, 2]
        target = positions[:, self.history_len:, :, :]  # [B, T, O, 2]
        gt_positions = positions[:, :self.history_len, :, :]  # [B, H, O, 2]
        with torch.no_grad():
            features = self.feature_extractor(m_frame, m_masks, gt_positions)
            predictions = self.predictor(features, target[:, 0])
            for t_pred in predictions[0]:
                for i, prediction in enumerate(t_pred):
                    x, y = prediction[0] * 160, prediction[1] * 210
                    frame = cv2.circle(frame, (int(x), int(y)), 1, color_map[i], 1)
            if self.show_prediction.get() == 2:
                for t_pred in target[0]:
                    for i, prediction in enumerate(t_pred):
                        x, y = prediction[0] * 160, prediction[1] * 210
                        frame = cv2.circle(frame, (int(x), int(y)), 1, color_map[i], 1)
        return frame
