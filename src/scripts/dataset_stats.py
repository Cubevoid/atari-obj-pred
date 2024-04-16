import sys
import numpy as np
from matplotlib import pyplot as plt

from src.data_collection.data_loader import DataLoader


def main(game: str) -> None:
    print(f"Loading data for {game}...")
    dl_sam = DataLoader(game, "SAM", 32, 4)
    dl_fastsam = DataLoader(game, "FastSAM-x", 32, 4)
    num_frames = min(len(dl_sam.frames), len(dl_fastsam.frames))
    objects_sam = dl_sam.object_types[:num_frames]
    objects_fastsam = dl_fastsam.object_types[:num_frames]
    print(f"Number of frames: {num_frames}")
    obj_per_frame_sam = np.count_nonzero(objects_sam, axis=1)
    obj_per_frame_fastsam = np.count_nonzero(objects_fastsam, axis=1)
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


if __name__ == "__main__":
    main(sys.argv[1])
