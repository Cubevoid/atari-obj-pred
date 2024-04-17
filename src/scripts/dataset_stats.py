import sys
import numpy as np
from numpy import typing as npt
from matplotlib import pyplot as plt

from src.data_collection.data_loader import DataLoader


def calculate_iou(boxes1, boxes2):
    """
    Calculate IoU (Intersection over Union) between corresponding bounding boxes.
    boxes1 and boxes2 should have the format (T, O, 4), where T is the number of timesteps,
    O is the number of objects, and 4 represents (x2, y2, x1, y1).
    """
    x1_1, y1_1, x2_1, y2_1 = np.split(boxes1, 4, axis=-1)
    x1_2, y1_2, x2_2, y2_2 = np.split(boxes2, 4, axis=-1)
    x1_1, x2_1 = np.minimum(x1_1, x2_1), np.maximum(x1_1, x2_1)
    y1_1, y2_1 = np.minimum(y1_1, y2_1), np.maximum(y1_1, y2_1)
    x1_2, x2_2 = np.minimum(x1_2, x2_2), np.maximum(x1_2, x2_2)
    y1_2, y2_2 = np.minimum(y1_2, y2_2), np.maximum(y1_2, y2_2)
    # Calculate intersection coordinates
    x1 = np.maximum(x1_1, x1_2)
    y1 = np.maximum(y1_1, y1_2)
    x2 = np.minimum(x2_1, x2_2)
    y2 = np.minimum(y2_1, y2_2)
    # Swap coordinates where necessary
    x1, x2 = np.minimum(x1, x2), np.maximum(x1, x2)
    y1, y2 = np.minimum(y1, y2), np.maximum(y1, y2)
    intersection_area = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    boxes1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    boxes2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = boxes1_area + boxes2_area - intersection_area
    iou = np.where(union_area > 0, intersection_area / union_area, 0)
    return np.clip(iou, 0, 1)


def main(game: str) -> None:
    print(f"Loading data for {game}...")
    dl_sam = DataLoader(game, "SAM", 32, 4)
    dl_fastsam = DataLoader(game, "FastSAM-x", 32, 4, max_data=len(dl_sam.frames))
    num_frames = min(len(dl_sam.frames), len(dl_fastsam.frames))
    print(f"Number of frames: {num_frames}")
    objects_sam = dl_sam.object_bounding_boxes[:num_frames]  # [T, O, 4]
    objects_fastsam = dl_fastsam.object_bounding_boxes[:num_frames]  # [T, O, 4]
    obj_per_frame_sam = np.count_nonzero(objects_sam[:, :, :2].sum(-1), axis=-1)
    obj_per_frame_fastsam = np.count_nonzero(objects_fastsam[:, :, :2].sum(-1), axis=-1)
    print(f"Average number of objects per frame for SAM: {np.mean(obj_per_frame_sam)}")
    print(f"Average number of objects per frame for FastSAM-x: {np.mean(obj_per_frame_fastsam)}")
    # Make plot of number of objects per frame
    plt.figure()
    plt.plot(obj_per_frame_sam, label="SAM")
    plt.plot(obj_per_frame_fastsam, label="FastSAM-x")
    plt.xlabel("Frame")
    plt.ylabel("Number of objects")
    plt.legend()
    plt.title(f"SAM vs FastSAM-x: Number of objects per frame for {game}")
    plt.savefig(f"{game}_SAM_vs_FastSAM.png")

    # Calculate IoU between FastSAM and SAM bboxes
    iou = calculate_iou(objects_sam, objects_fastsam).squeeze(-1)
    num_obj = np.maximum(obj_per_frame_sam, obj_per_frame_fastsam)  # per frame
    ious = [np.mean(iou[i, : num_obj[i]]) for i in range(min(num_frames, 50))]
    # Make new plot of IoUs
    plt.figure()
    plt.plot(ious, label="Mean IoU")
    plt.xlabel("Frame")
    plt.ylabel("IoU")
    plt.legend()
    plt.title(f"Mean IoU between SAM and FastSAM-x bboxes for {game}")
    plt.savefig(f"{game}_IoU.png")


if __name__ == "__main__":
    main(sys.argv[1])
