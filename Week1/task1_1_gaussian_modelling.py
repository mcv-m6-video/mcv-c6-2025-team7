from pathlib import Path
from utils import estimate_background, segment_foreground
from typing import Optional

def gaussian_modelling(
    video_path: str,
    percent: float = 0.25,
    alpha: float = 2.5,
    use_median: bool = False,
    output_name: Optional[str] = None,  # Updated for Python 3.9
    show_video: bool = False
) -> None:
    """Processes a video using Gaussian modeling for background subtraction.

    This function estimates the background of a video using a Gaussian model
    (mean or median-based) and segments the foreground using a thresholded 
    absolute difference approach. The processed video is saved in the `Output_Videos` 
    folder with a custom or automatically generated filename.

    Args:
        video_path (str): Path to the input video file.
        percent (float, optional): The fraction (0 to 1) of the total frames 
            used for background estimation (default is 0.25).
        alpha (float, optional): The multiplier for the standard deviation 
            in the foreground detection threshold (default is 2.5).
        use_median (bool, optional): If True, uses the median instead of the mean 
            for background estimation (default is False).
        output_name (str | None, optional): Custom name for the output video file. 
            If None, the name is based on the input video filename (default is None).
        show_video (bool, optional): If True, displays the segmented foreground in real-time (default is False).

    Returns:
        None: The function does not return a value but saves the processed video.

    Example:
        >>> gaussian_modelling("video.mp4", percent=0.2, alpha=3.0, use_median=True, output_name="processed_video", show_video=True)

    Notes:
        - The output video is stored in the `Output_Videos` directory.
        - If background estimation fails, the function prints an error message and exits.
        - Ensure the video file is accessible and in a supported format.
    """
    
    output_folder = "Output_Videos"
    Path(output_folder).mkdir(parents=True, exist_ok=True)  # Create the output folder if it doesn't exist
    
    if not output_name: 
        output_name = Path(video_path).stem + "_output"
    
    output_video_path = Path(output_folder) / f"{output_name}.avi"  # Define the output video path
    
    mean_bg, std_bg, num_bg_frames = estimate_background(video_path, percent, use_median)  # Estimate background

    if mean_bg is None or std_bg is None:  # Error in background estimation
        print("Background estimation failed. Check the video path and format.")
    else:
        # Segment foreground and save to video
        segment_foreground(
            video_path,
            mean_bg,
            std_bg,
            num_bg_frames=535,  # Start at frame 536 to sincronize
            alpha=alpha,
            output_path=output_video_path,
            bbox_output_json="detected_bounding_boxes.json",
            show_video=show_video
        )


if __name__ == "__main__":
    video_path = Path("AICity_data") / "train" / "S03" / "c010" / "vdo.avi"
    assert video_path.exists(), f"{video_path} doesn't exists!"

    # Single Gaussian Modelling 
    gaussian_modelling(video_path.as_posix(), 
                       percent=0.25, 
                       alpha=3.5,  # It was at 2.5, I have augmented it
                       use_median=False, 
                       output_name="task_1_1_mean_alpha2.5-10",  
                       show_video=False) 
