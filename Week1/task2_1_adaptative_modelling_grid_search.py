from pathlib import Path

import numpy as np

from utils import adaptive_segment_foreground, estimate_background


def adaptative_modelling(
        video_path: str,
        percent: float = 0.25,
        alpha: float = 2.5,
        rho: float = 0.01,
        use_median: bool = False,
        output_name: str | None = None,
        show_video: bool = False
) -> None:
    """ TODO """

    output_folder = Path("Output_Videos") / "AdaptiveModelling"
    output_folder.mkdir(parents=True, exist_ok=True)  # Create the output folder if it doesn't exist

    if not output_name:
        output_name = Path(video_path).stem + "_output"

    output_video_path = output_folder / f"{output_name}.avi"  # Define the output video path

    mean_bg, std_bg, num_bg_frames = estimate_background(video_path, percent, use_median)  # Estimate background

    if mean_bg is None or std_bg is None:  # Error in background estimation
        print("Background estimation failed. Check the video path and format.")
    else:
        # Segment foreground and save to video
        adaptive_segment_foreground(video_path, mean_bg, std_bg, num_bg_frames, alpha, rho, output_video_path.as_posix(), show_video)


if __name__ == "__main__":
    video_path = Path("AICity_data") / "train" / "S03" / "c010" / "vdo.avi"
    assert video_path.exists(), f"{video_path} doesn't exists!"

    parameter_search = {
        "alpha": list(np.arange(0, 0.7, 0.1)),
        "rho": list(np.arange(0, 10, 1)),
    }

    for alpha in parameter_search["alpha"]:
        for rho in parameter_search["rho"]:
            adaptative_modelling(
                video_path.as_posix(),
                percent=0.25,
                alpha=alpha,
                rho=rho,
                use_median=False,
                output_name=f"task_2_1_mean_alpha{alpha}_rho{rho}",
                show_video=False
            )
