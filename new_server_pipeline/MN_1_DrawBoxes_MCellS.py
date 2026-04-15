"""
LF_1_DrawBoxes_MCellS.py

Interactive tool for selecting rectangular regions on binary frame data.
Load a .bin video frame, draw bounding boxes around wells of interest,
and save the region coordinates as .npy files.

Usage:
    python LF_1_DrawBoxes_MCellS.py /path/to/dat.bin
    python LF_1_DrawBoxes_MCellS.py /path/to/dat.bin --width 2048 --height 2048
    python LF_1_DrawBoxes_MCellS.py  # launches interactive prompt
    # use: ./data/Raw_data/2025_10_15_10_20_58_trial_1_TC/dat.bin as example

Functions can also be imported:
    from LF_1_DrawBoxes_MCellS import load_frame, select_regions, save_regions
"""

import numpy as np
import os
import argparse
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle


def load_frame(filepath, width=2048, height=2048):
    """Load the first frame from a raw binary video file.

    Parameters
    ----------
    filepath : str or Path
        Path to the .bin file.
    width : int
        Frame width in pixels.
    height : int
        Frame height in pixels.

    Returns
    -------
    np.ndarray
        2D array of shape (height, width) with dtype uint8.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    frame_size = width * height
    with open(filepath, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint8, count=frame_size)

    if len(data) < frame_size:
        raise ValueError(
            f"File too small for {width}x{height}. "
            f"Expected {frame_size} bytes, got {len(data)}."
        )

    return data.reshape((height, width))


def select_regions(frame):
    """Display a frame and let the user draw rectangular regions interactively.

    Click and drag to create a region. Close the window when done.

    Parameters
    ----------
    frame : np.ndarray
        2D grayscale image array.

    Returns
    -------
    list[list[int]]
        Each element is [x_min, y_min, x_max, y_max].
    """
    regions = []
    rectangles = []

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(frame, cmap='gray')
    ax.set_title('Click and drag to select regions. Close window when done.')

    def on_select(eclick, erelease):
        if eclick.xdata is None or erelease.xdata is None:
            return

        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)

        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)

        regions.append([x_min, y_min, x_max, y_max])

        rect = Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            fill=False, edgecolor='red', linewidth=2,
        )
        ax.add_patch(rect)
        rectangles.append(rect)

        print(f"  Region {len(regions)}: ({x_min}, {y_min}) to ({x_max}, {y_max})")
        fig.canvas.draw_idle()

    selector = RectangleSelector(
        ax, on_select,
        useblit=True,
        button=[1],
        minspanx=5, minspany=5,
        spancoords='pixels',
        interactive=True,
    )

    # Keep a reference so the selector isn't garbage-collected
    fig._region_selector = selector

    plt.show(block=True)

    return regions


def save_regions(regions, bin_path, frame_shape):
    """Save selected regions as a .npy file next to the source bin file.

    Parameters
    ----------
    regions : list[list[int]]
        Region coordinates from select_regions().
    bin_path : str or Path
        Path to the original .bin file (used to derive the save path).
    frame_shape : tuple[int, int]
        (height, width) of the frame.

    Returns
    -------
    Path
        The path where the .npy file was saved.
    """
    if not regions:
        print("No regions to save.")
        return None

    bin_path = Path(bin_path)
    parent_folder_name = bin_path.parent.name
    h, w = frame_shape
    filename = f'regions_{parent_folder_name}_{w}x{h}.npy'
    save_path = bin_path.parent / filename

    regions_array = np.array(regions)
    np.save(save_path, regions_array, allow_pickle=False)

    print(f"Saved {len(regions)} regions to {save_path}")
    for i, region in enumerate(regions, 1):
        print(f"  Region {i}: ({region[0]}, {region[1]}) to ({region[2]}, {region[3]})")

    return save_path


def run(bin_path, width=2048, height=2048):
    """Full pipeline: load frame, select regions interactively, save.

    Parameters
    ----------
    bin_path : str or Path
        Path to the .bin file.
    width : int
        Frame width in pixels.
    height : int
        Frame height in pixels.

    Returns
    -------
    list[list[int]]
        The selected regions.
    """
    print(f"Loading frame from {bin_path} ({width}x{height})...")
    frame = load_frame(bin_path, width, height)
    print("Frame loaded. Draw regions, then close the window.")

    regions = select_regions(frame)

    if regions:
        save_regions(regions, bin_path, frame.shape)
    else:
        print("No regions were selected.")

    return regions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Draw bounding-box regions on a raw .bin video frame.',
    )
    parser.add_argument(
        'bin_path', nargs='?', default=None,
        help='Path to the .bin file. If omitted, you will be prompted.',
    )
    parser.add_argument('--width', type=int, default=2048, help='Frame width (default: 2048)')
    parser.add_argument('--height', type=int, default=2048, help='Frame height (default: 2048)')
    args = parser.parse_args()

    if args.bin_path is None:
        args.bin_path = input("Enter path to .bin file: ").strip()

    run(args.bin_path, args.width, args.height)
