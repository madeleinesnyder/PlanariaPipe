"""
LF_2_Make_JPGs.py - Longform WormShadow batch and anonymization compatible

This script takes regions files (created in Notebook 1) and the original dat.bin
footage to create:
  * Segmented or continuous .jpeg and .mp4 footage of the bin files.
  * Labels output footage using original session name or with anonymous codenames.

The jpeg series produced by this script are used to create masks of worms in
Notebook 3. The anonymous footage created by this script can be fed to the Blind
Scoring Interface for manual scoring of anonymous footage.

This script requires the environment "planarian_segmentation", as do the next
several notebooks.

NOTE: This script is functionally equivalent to
LF_2_WormShadow_20251211v1_LfwithAnon.ipynb. It was reconfigured and
restructured for batch processing, but the output videos and so on are the same.
"""

import os
import time
import cv2
import numpy as np
import glob
import gc
import pandas as pd
import random
import csv
import argparse
from pathlib import Path
from datetime import datetime


# ============================================================================
# SESSION LIST FOR BATCH PROCESSING
# ============================================================================

SESSIONS_TO_PROCESS = [
    '2025_10_17_14_54_47_trial_1_TP',
    '2025_10_17_14_45_34_trial_1_TP'
]

# ============================================================================
# GLOBAL PROCESSING FLAGS (apply to all sessions in batch)
# ============================================================================

LONG_OR_SHORT = 'short' #'long'        # Options: 'long' or 'short', indicates whether the OUTPUT video is one long video or a video per trial
ANONYMIZATION = True                   # Options: True or False
SEPARATE_CS_ON_OFF_BACKGROUNDS = True  # True: CSon/CSoff, False: single averaged
CROPPED_VIDEOS = True                 # True means the SOURCE videos have the ITI cut out; False means the original videos were continuous/longform

# ============================================================================
# CAMERA AND TIMING PARAMETERS
# ============================================================================

width = 2048
height = 2048
FRAME_RATE = 10
PRE_SHIFT_FRAMES = 5
CUT_DELAY_AFTER_SHOCK = 2.15
CUT_DELAY_BEFORE_LIGHT = 2.0
TP_TOTAL_FRAMES = 274  # Total frames for TP trial videos
TC_TOTAL_FRAMES = 274


# ============================================================================
# DIRECTORY PATHS
# ============================================================================

HOLYLABS = "data" # "/n/holylabs/gershman_lab/Users/zkelso/Raw_data"
# DLC_PROJECT_PATH = '/n/holylabs/LABS/gershman_lab/Users/zkelso/DLC_Projects/WormTracking'
DLC_PROJECT_PATH = "data/temporary_jpgs"

# Anonymization file paths
ANONYMIZATION_DICT_PATH = 'Anonymization_tools/anonymization_dictionary.csv'
LOOKUP_CSV_PATH = 'Anonymization_tools/anonymization_lookup.csv'
EXPERIMENT_LOG = 'Anonymization_tools/experiment_log.csv'

# ============================================================================
# BLOCK-TO-TRIALS MAPPING
# ============================================================================

TP_BLOCK_TO_TRIALS = {    # Which TP trials to put into trial_definitions.csv
    1: [1, 2],
    2: [1],
    3: [1, 2], # Formerly just [2], but is really [1, 2] watching video
    4: [2],
    5: [1, 3],
    6: [1],
    7: [1, 3],
    8: [2]
}

TC_CROPPED_TRIAL_FRAMES_SHORT = {   # Indicated by coming on of indicators; buffer periods not included. Last frame is frame before shock comes on. 
    1: [23, 292], 
    2: [363, 632],
    3: [703, 972],
}

TP_CROPPED_TRIAL_FRAMES_SHORT = {       # Indicated by coming on of indicators; buffer periods not included.
    1: [23, 292],                     # These numbers would need to be confirmed, but also, we're just doing TC videos for now. So I won't confirm/deal with this until needed.
    2: [363, 632],
    3: [703, 972],
}


# ============================================================================
# MODULE-LEVEL GLOBALS (updated by functions during processing)
# ============================================================================

binfile_to_segment = None
COORD_FOLDER = None
SESSION_TYPE = None
ALL_TRIAL_DEFINITIONS = None


# ============================================================================
# CORE PROCESSING FUNCTIONS (Block 2)
# ============================================================================

def update_session_context(session_name):
    """Update session-specific variables for the current session"""
    global binfile_to_segment, COORD_FOLDER, SESSION_TYPE
    
    binfile_to_segment = session_name
    COORD_FOLDER = os.path.join(HOLYLABS, 'Raw_data', binfile_to_segment)
    
    # Detect session type from name
    if binfile_to_segment.endswith('_TC'):
        SESSION_TYPE = 'TC'
    elif binfile_to_segment.endswith('_TP'):
        SESSION_TYPE = 'TP'
    else:
        SESSION_TYPE = 'TC'
    
    return SESSION_TYPE


def calculate_trial_definitions_from_timestamps():
    """
    Calculate trial frame ranges from stim_extra.csv timestamps.
    
    For TC sessions: Uses light_on to shock_on
    For TP sessions: Uses fixed 274-frame window from light_on
    
    ALL trials found in CSV are processed (no filtering).
    
    Returns:
        dict: {trial_num: (start_frame, end_frame)} or None if calculation fails
    """
    
    print("  Calculating trial definitions from timestamps...")
    
    # Load stimulus timing data
    stim_csv_path = os.path.join(HOLYLABS, 'Raw_data', binfile_to_segment, 'stim_extra.csv')

    
    if not os.path.exists(stim_csv_path):
        print(f"    WARNING: Stimulus file not found: {stim_csv_path}")
        return None
    
    try:
        df = pd.read_csv(stim_csv_path)
        
        # Extract trials with valid light_on timestamps
        trials = []
        for idx, row in df.iterrows():
            if not pd.isna(row.get('white_light_on')):
                trial_info = {
                    'trial_num': idx + 1,
                    'light_on': row['white_light_on']
                }
                
                # For TC sessions, we also need shock timing
                if SESSION_TYPE == 'TC' and not pd.isna(row.get('shock_on')):
                    trial_info['shock_on'] = row['shock_on']
                    trial_info['shock_off'] = row.get('shock_off', None)
                    trials.append(trial_info)
                elif SESSION_TYPE == 'TP':
                    # TP only needs light_on
                    trials.append(trial_info)
        
        if not trials:
            raise ValueError("No valid trials found in stimulus data")
        
        # Infer video start time from first trial
        first_light = trials[0]['light_on']
        first_light_frame = 23  # Empirically verified
        video_start_ms = first_light - (first_light_frame * 1000.0 / FRAME_RATE)
        
        # Calculate trial frame windows
        calculated_trials = {}
        
        if SESSION_TYPE == 'TC':
            # TC: Use light_on to shock_on
            for trial in trials:
                trial_num = trial['trial_num']
                
                light_elapsed_ms = trial['light_on'] - video_start_ms
                light_frame = int(light_elapsed_ms / 1000.0 * FRAME_RATE)
                trial_start = light_frame - PRE_SHIFT_FRAMES
                
                shock_elapsed_ms = trial['shock_on'] - video_start_ms
                shock_frame = int(shock_elapsed_ms / 1000.0 * FRAME_RATE)
                trial_end = shock_frame - 1
                
                calculated_trials[trial_num] = (trial_start, trial_end)
        
        else:  # SESSION_TYPE == 'TP'
            # TP: Fixed window from light_on
            for trial in trials:
                trial_num = trial['trial_num']
                
                light_elapsed_ms = trial['light_on'] - video_start_ms
                light_frame = int(light_elapsed_ms / 1000.0 * FRAME_RATE)
                trial_start = light_frame - PRE_SHIFT_FRAMES
                trial_end = trial_start + TP_TOTAL_FRAMES - 1
                
                calculated_trials[trial_num] = (trial_start, trial_end)
        
        print(f"    Found {len(calculated_trials)} trials")
        return calculated_trials
        
    except Exception as e:
        print(f"    ERROR calculating trial definitions: {e}")
        return None


def calculate_trial_definitions_from_dictionary():
    if SESSION_TYPE == 'TC':
        raw_dict = TC_CROPPED_TRIAL_FRAMES_SHORT
    else:
        raw_dict = TP_CROPPED_TRIAL_FRAMES_SHORT
    
    calculated_trials = {}
    for trial_num, (dict_start, dict_end) in raw_dict.items():
        trial_start = dict_start - PRE_SHIFT_FRAMES  # Apply 5-frame pre-buffer
        trial_end = dict_end                          # End unchanged
        calculated_trials[trial_num] = (trial_start, trial_end)
    
    return calculated_trials


def get_tp_protocol_trial_labels(num_csv_trials):
    """
    Get protocol trial numbers for labeling TP videos in anonymization lookup.
    Does NOT filter trials - only provides labels for anonymization.
    
    Args:
        num_csv_trials: Number of trials found in stim_extra.csv
    
    Returns:
        list: Protocol trial numbers (e.g., [1, 3] for Block 7)
        
    Raises:
        ValueError: If number of labels doesn't match number of CSV trials
    """
    
    if SESSION_TYPE != 'TP':
        # TC sessions use sequential numbering
        return list(range(1, num_csv_trials + 1))
    
    # Read experiment log to get Block number
    if not os.path.exists(EXPERIMENT_LOG):
        raise FileNotFoundError(f"Experiment log not found: {EXPERIMENT_LOG}")
    
    exp_log_df = pd.read_csv(EXPERIMENT_LOG)
    session_row = exp_log_df[exp_log_df['Data_Folder'] == binfile_to_segment]
    
    if len(session_row) == 0:
        raise ValueError(f"Session '{binfile_to_segment}' not found in experiment log")
    
    block_number = int(session_row['Block'].iloc[0])
    
    if block_number not in TP_BLOCK_TO_TRIALS:
        raise ValueError(f"Block {block_number} not found in TP_BLOCK_TO_TRIALS dictionary")
    
    protocol_trial_labels = TP_BLOCK_TO_TRIALS[block_number]
    
    # VALIDATE: Number of labels must match number of CSV trials
    if len(protocol_trial_labels) != num_csv_trials:
        raise ValueError(
            f"TRIAL LABELING MISMATCH for session '{binfile_to_segment}':\n"
            f"  Block {block_number} expects {len(protocol_trial_labels)} trials: {protocol_trial_labels}\n"
            f"  But stim_extra.csv contains {num_csv_trials} trials\n"
            f"  Check TP_BLOCK_TO_TRIALS dictionary or Block assignment in experiment log"
        )
    
    print(f"    Block {block_number}: labeling {num_csv_trials} trials as {protocol_trial_labels}")
    return protocol_trial_labels


def validate_trial_ranges(data_shape, trial_definitions):
    """Check if trial frame ranges are valid for the video length"""
    total_frames = data_shape[0]
    
    for trial_num, (start_frame, end_frame) in trial_definitions.items():
        if end_frame >= total_frames:
            raise ValueError(f"Trial {trial_num} end frame ({end_frame}) exceeds video length ({total_frames} frames)")
        if start_frame < 0:
            raise ValueError(f"Trial {trial_num} start frame ({start_frame}) is negative")
        if start_frame > end_frame:
            raise ValueError(f"Trial {trial_num} start frame ({start_frame}) > end frame ({end_frame})")


def make_DLC_videos_from_jpegs(folder_name, SOURCE_FRAMES_DLC):
    """Create MP4 from existing JPEGs in folder"""
    
    images = sorted(glob.glob(os.path.join(SOURCE_FRAMES_DLC, "*.jpeg")))
    
    if not images:
        print(f"      WARNING: No JPEG files found at {os.path.join(SOURCE_FRAMES_DLC, '*.jpeg')}")
        return
    
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_filename = f"{folder_name}.mp4"
    video_path = os.path.join(DLC_PROJECT_PATH, 'videos', video_filename)
    
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    
    out = cv2.VideoWriter(video_path, fourcc, 10, (width, height))
    
    total_frames = len(images)
    
    for i, image_path in enumerate(images):
        frame = cv2.imread(image_path)
        out.write(frame)
        
        # Overwriting progress bar
        if total_frames > 10:
            progress = ((i + 1) / total_frames) * 100
            print(f"\r      Creating MP4: {progress:.0f}% [{i+1}/{total_frames}]", end='', flush=True)
    
    out.release()
    if total_frames > 10:
        print()  # New line after progress bar


def create_trial_specific_backgrounds():
    """
    Create two background images per coordinate set:
    - CSon: Average of frames during trial periods
    - CSoff: Average of frames during inter-trial periods
    """
    
    print("    Creating CS on/off backgrounds...")
    
    # Load regions file
    regions_pattern = os.path.join(COORD_FOLDER, "regions*")
    regions_files = glob.glob(regions_pattern)
    
    if not regions_files:
        print(f"      ERROR: No regions file found at {regions_pattern}")
        return
    
    cropped_dims = np.load(regions_files[0])
    
    # Load full video
    bin_pattern = os.path.join(HOLYLABS, 'Raw_data', binfile_to_segment, 'dat.bin')
    bin_files = glob.glob(bin_pattern)
    
    if not bin_files:
        print(f"      ERROR: No .bin file found")
        return
    
    SOURCE_VIDEO = bin_files[0]
    
    with open(SOURCE_VIDEO, 'rb') as f:
        data_ = np.fromfile(f, dtype=np.uint8)
        
        frame_size = width * height
        complete_frames = data_.shape[0] // frame_size
        data_ = data_[:complete_frames * frame_size]
        data_ = data_.reshape(complete_frames, height, width)
    
    # Ensure the background output directory exists
    background_output_dir = os.path.join(HOLYLABS, "Calibration_images", "Full_videos")
    os.makedirs(background_output_dir, exist_ok=True)
    
    # Determine CSon and CSoff frame indices using ALL trials
    CSon_frames = []
    for trial_num, (start_frame, end_frame) in ALL_TRIAL_DEFINITIONS.items():
        CSon_frames.extend(range(start_frame, end_frame + 1))
    
    CSon_frames = sorted(set(CSon_frames))
    all_frames = set(range(data_.shape[0]))
    CSoff_frames = sorted(all_frames - set(CSon_frames))
    
    # Process each coordinate set
    for coord_idx, coords in enumerate(cropped_dims):
        # Define crop boundaries
        x_start = int(min(coords[0], coords[2]))
        x_end = int(max(coords[0], coords[2]))
        y_start = int(min(coords[1], coords[3]))
        y_end = int(max(coords[1], coords[3]))
        
        # Ensure within bounds
        x_start = max(0, x_start)
        y_start = max(0, y_start)
        x_end = min(data_.shape[2], x_end)
        y_end = min(data_.shape[1], y_end)
        
        # Create CSon background
        CSon_sum = np.zeros((y_end - y_start, x_end - x_start), dtype=np.float64)
        for frame_idx in CSon_frames:
            frame = data_[frame_idx]
            cropped = frame[y_start:y_end, x_start:x_end]
            CSon_sum += cropped.astype(np.float64)
        CSon_background = (CSon_sum / len(CSon_frames)).astype(np.uint8)
        
        # Create CSoff background
        CSoff_sum = np.zeros((y_end - y_start, x_end - x_start), dtype=np.float64)
        for frame_idx in CSoff_frames:
            frame = data_[frame_idx]
            cropped = frame[y_start:y_end, x_start:x_end]
            CSoff_sum += cropped.astype(np.float64)
        CSoff_background = (CSoff_sum / len(CSoff_frames)).astype(np.uint8)
        
        # Save backgrounds
        coord_string = f"{coords[0]}_{coords[1]}_{coords[2]}_{coords[3]}"
        
        CSon_filename = f"{binfile_to_segment}_regions_{coord_string}_CSon.jpg"
        CSoff_filename = f"{binfile_to_segment}_regions_{coord_string}_CSoff.jpg"
        
        CSon_path = os.path.join(background_output_dir, CSon_filename)
        CSoff_path = os.path.join(background_output_dir, CSoff_filename)
        
        CSon_bgr = cv2.cvtColor(CSon_background, cv2.COLOR_GRAY2BGR)
        CSoff_bgr = cv2.cvtColor(CSoff_background, cv2.COLOR_GRAY2BGR)
        
        cv2.imwrite(CSon_path, CSon_bgr)
        cv2.imwrite(CSoff_path, CSoff_bgr)
    
    # Save trial definitions as CSV
    trial_csv_path = os.path.join(HOLYLABS, 'Raw_data', binfile_to_segment, "trial_definitions.csv")
    os.makedirs(os.path.dirname(trial_csv_path), exist_ok=True)   
    trial_df = pd.DataFrame([
        {'trial_num': trial_num, 'start_frame': start_frame, 'end_frame': end_frame}
        for trial_num, (start_frame, end_frame) in ALL_TRIAL_DEFINITIONS.items()
    ])
    trial_df.to_csv(trial_csv_path, index=False)
    print(f"File trial_definitions.csv saved to {trial_csv_path}.")
    
    # Clean up
    del data_
    gc.collect()
    
    print(f"      Created {len(cropped_dims)} CS on/off background pairs")


def create_simple_averaged_background():
    """
    Create single averaged background from ALL frames.
    Faster alternative to CS on/off backgrounds.
    """
    
    print("    Creating simple averaged backgrounds...")
    
    # Load regions file
    regions_pattern = os.path.join(COORD_FOLDER, "regions*")
    regions_files = glob.glob(regions_pattern)
    
    if not regions_files:
        print(f"      ERROR: No regions file found at {regions_pattern}")
        return
    
    cropped_dims = np.load(regions_files[0])
    
    # Load full video
    bin_pattern = os.path.join(HOLYLABS, 'Raw_data', binfile_to_segment, '*dat.bin')
    bin_files = glob.glob(bin_pattern)
    
    if not bin_files:
        print(f"      ERROR: No .bin file found")
        return
    
    SOURCE_VIDEO = bin_files[0]
    
    with open(SOURCE_VIDEO, 'rb') as f:
        data_ = np.fromfile(f, dtype=np.uint8)
        
        frame_size = width * height
        complete_frames = data_.shape[0] // frame_size
        data_ = data_[:complete_frames * frame_size]
        data_ = data_.reshape(complete_frames, height, width)
    
    # Ensure the background output directory exists
    background_output_dir = os.path.join(HOLYLABS, "Calibration_images", "Full_videos")
    os.makedirs(background_output_dir, exist_ok=True)
    
    # Process each coordinate set
    for coord_idx, coords in enumerate(cropped_dims):
        # Define crop boundaries
        x_start = int(min(coords[0], coords[2]))
        x_end = int(max(coords[0], coords[2]))
        y_start = int(min(coords[1], coords[3]))
        y_end = int(max(coords[1], coords[3]))
        
        # Ensure within bounds
        x_start = max(0, x_start)
        y_start = max(0, y_start)
        x_end = min(data_.shape[2], x_end)
        y_end = min(data_.shape[1], y_end)
        
        # Create averaged background from all frames
        avg_sum = np.zeros((y_end - y_start, x_end - x_start), dtype=np.float64)
        
        for frame_idx in range(data_.shape[0]):
            frame = data_[frame_idx]
            cropped = frame[y_start:y_end, x_start:x_end]
            avg_sum += cropped.astype(np.float64)
            
            # Overwriting progress bar
            if (frame_idx + 1) % 100 == 0:
                progress = ((frame_idx + 1) / data_.shape[0]) * 100
                print(f"\r      Processing coord {coord_idx+1}/{len(cropped_dims)}: {progress:.0f}%", end='', flush=True)
        
        print()  # New line after progress bar
        
        avg_background = (avg_sum / data_.shape[0]).astype(np.uint8)
        
        # Save background
        coord_string = f"{coords[0]}_{coords[1]}_{coords[2]}_{coords[3]}"
        avg_filename = f"{binfile_to_segment}_regions_{coord_string}_AVG.jpg"
        avg_path = os.path.join(background_output_dir, avg_filename)
        
        avg_bgr = cv2.cvtColor(avg_background, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(avg_path, avg_bgr)
    
    # Clean up
    del data_
    gc.collect()
    
    print(f"      Created {len(cropped_dims)} averaged backgrounds")


def create_backgrounds():
    """Wrapper to call appropriate background creation function based on flag"""
    if SEPARATE_CS_ON_OFF_BACKGROUNDS:
        create_trial_specific_backgrounds()
    else:
        create_simple_averaged_background()


def find_missing_data(probe_id):
    """
    Recursively searches HOLYLABS for the session ID. This is called when a bin file can't be found. Gives print statements identifying file location.
    """
    # Define search paths
    SEARCH_PATHS = {
        "HOLYLABS": "data"
    }

    for location_name, path in SEARCH_PATHS.items():
        print(f"\tSearching in {location_name} for {probe_id}...")
        found_something = False

        # os.walk allows us to look recursively through every subfolder
        for root, dirs, files in os.walk(path):
            # Check both files and directories
            for name in files + dirs:
                if probe_id in name:
                    full_path = os.path.join(root, name)
                    print(f"\t\tFOUND: {full_path}")
                    found_something = True

        if not found_something:
            print(f"\t\t{probe_id} not found in at least one location. Check if it's Holylabs.")


# ============================================================================
# ANONYMIZATION FUNCTIONS (Block 3)
# ============================================================================

def load_anonymization_dictionary(dict_path):
    """Load the word dictionary for anonymous name generation"""
    
    with open(dict_path) as f:
        reader = list(csv.DictReader(f))
        
        valid_adjectives = [
            row['Adjective'] for row in reader 
            if row['Adjective'] and 
               row['Adjective'].strip() != '' and
               row['Adjective'] != 'None'
        ]
        
        valid_nouns = [
            row['Noun'] for row in reader 
            if row['Noun'] and 
               row['Noun'].strip() != '' and
               row['Noun'] != 'None'
        ]
    
    return valid_adjectives, valid_nouns


def get_all_existing_names(lookup_csv_path):
    """Get all existing anonymous names from the lookup CSV"""
    existing_names = set()
    
    if os.path.exists(lookup_csv_path):
        try:
            df = pd.read_csv(lookup_csv_path)
            if 'Anonymous_Name' in df.columns and len(df) > 0:
                all_names = df['Anonymous_Name'].dropna().tolist()
                existing_names.update(all_names)
        except Exception as e:
            pass
    
    return existing_names


def check_name_with_suffix(base_name, existing_names):
    """
    Check if base_name exists, if so add suffix _2, _3, etc.
    Returns the first available name with suffix if needed
    """
    
    if base_name not in existing_names:
        return base_name
    
    suffix_num = 2
    max_suffix_attempts = 1000
    
    while suffix_num <= max_suffix_attempts:
        candidate_name = f"{base_name}_{suffix_num}"
        
        if candidate_name not in existing_names:
            return candidate_name
        
        suffix_num += 1
    
    raise ValueError(f"Could not find available suffix for {base_name} after {max_suffix_attempts} attempts")


def generate_anonymous_video_name(adjectives, nouns, used_names=set(), lookup_csv_path=None):
    """
    Generate unique Video_Adj_Adj_Noun name with duplicate protection
    Checks both in-memory used_names and the lookup CSV for duplicates
    """
    
    # Get all existing names from CSV for comprehensive duplicate checking
    if lookup_csv_path:
        csv_existing_names = get_all_existing_names(lookup_csv_path)
        all_existing_names = used_names.union(csv_existing_names)
    else:
        all_existing_names = used_names
    
    max_attempts = 1000
    for attempt in range(max_attempts):
        # Generate base name
        adj1, adj2 = random.sample(adjectives, 2)
        noun = random.choice(nouns)
        base_name = f"Video_{adj1}_{adj2}_{noun}"
        
        # Check for duplicates and add suffix if needed
        final_name = check_name_with_suffix(base_name, all_existing_names)
        
        # Add to used_names set for this session
        used_names.add(final_name)
        
        return final_name
    
    raise ValueError(f"Could not generate unique name after {max_attempts} attempts")


def create_lookup_csv(lookup_csv_path):
    """Create lookup CSV if it doesn't exist"""
    if not os.path.exists(lookup_csv_path):
        df = pd.DataFrame(columns=[
            'Anonymous_Name', 'Original_File', 'Box_Coordinates', 
            'Trial_Number', 'Trial_Frames', 'Session_Date', 'Trial_Type', 'Timestamp_Created'
        ])
        df.to_csv(lookup_csv_path, index=False)


def update_lookup_csv_with_trial(lookup_csv_path, anonymous_name, original_file, coords, trial_num, start_frame, end_frame):
    """Add new anonymization record with trial information to lookup CSV"""
    
    # Load existing lookup
    df = pd.read_csv(lookup_csv_path)
    
    # Extract metadata from filename
    parts = original_file.split('_')
    session_date = None
    trial_type = None
    
    if len(parts) >= 6:
        try:
            session_date = f"{parts[0]}_{parts[1]}_{parts[2]}"
            trial_type = parts[-1]
        except:
            pass
    
    coord_string = f"{coords[0]}_{coords[1]}_{coords[2]}_{coords[3]}"
    frame_range = f"{start_frame}-{end_frame}"
    
    new_record = {
        'Anonymous_Name': anonymous_name,
        'Original_File': original_file,
        'Box_Coordinates': coord_string,
        'Trial_Number': trial_num,
        'Trial_Frames': frame_range,
        'Session_Date': session_date,
        'Trial_Type': trial_type,
        'Timestamp_Created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
    df.to_csv(lookup_csv_path, index=False)


# ============================================================================
# ANONYMIZATION INITIALIZATION
# ============================================================================

def initialize_anonymization():
    """Initialize the anonymization system by loading dictionaries and existing names."""
    global adjectives, nouns, used_names

    if ANONYMIZATION:
        print("="*70)
        print("ANONYMIZATION SYSTEM INITIALIZATION")
        print("="*70)
        
        adjectives, nouns = load_anonymization_dictionary(ANONYMIZATION_DICT_PATH)
        create_lookup_csv(LOOKUP_CSV_PATH)
        
        # Load existing names to avoid duplicates (persists across all sessions)
        used_names = set()
        if os.path.exists(LOOKUP_CSV_PATH):
            try:
                existing_df = pd.read_csv(LOOKUP_CSV_PATH)
                if 'Anonymous_Name' in existing_df.columns and len(existing_df) > 0:
                    used_names.update(existing_df['Anonymous_Name'].dropna().tolist())
                    print(f"  Loaded {len(used_names)} existing anonymous names")
                else:
                    print("  Lookup CSV exists but is empty")
            except pd.errors.EmptyDataError:
                print("  Lookup CSV is empty")
            except Exception as e:
                print(f"  Warning: Error reading lookup CSV: {e}")
        
        # Show duplicate protection status
        total_possible_names = len(adjectives) * (len(adjectives) - 1) * len(nouns)
        collision_probability = len(used_names) / total_possible_names if total_possible_names > 0 else 0
        
        print(f"  Anonymization system initialized")
        print(f"  Possible unique names: {total_possible_names:,}")
        print(f"  Names already used: {len(used_names)}")
        print(f"  Collision probability: {collision_probability:.8f}")
        print("="*70)
        
    else:
        print("="*70)
        print("ANONYMIZATION DISABLED")
        print("="*70)
        print("  Using coordinate-based naming")
        print("="*70)
        adjectives, nouns, used_names = [], [], set()


# ============================================================================
# FLAG VALIDATION
# ============================================================================

def validate_flags():
    """Validate flag combinations and session list before processing."""
    print("\n" + "="*70)
    print("BATCH CONFIGURATION VALIDATION")
    print("="*70)

    # Check for invalid flag combinations
    if LONG_OR_SHORT == 'long' and ANONYMIZATION == True:
        raise ValueError(
            """ERROR: LONG_OR_SHORT='long' with ANONYMIZATION=True is not valid.
Do you really want to anonymize long videos? That doesn't really make sense...
... unless you know something I don't. Comment out this error raise if you want to proceed."""
        )

    if LONG_OR_SHORT == 'short' and ANONYMIZATION == False:
        raise ValueError(
            """ERROR: LONG_OR_SHORT='short' with ANONYMIZATION=False is not valid.
This is not a valid combination. Check your flags."""
        )


    if CROPPED_VIDEOS:
        UNCROPPED_DATE_STRINGS = ["2025_10_14", "2025_10_15", "2025_10_16", "2025_10_17"]  # These are the longform videos, so you shouldn't use cropped video methods for them.
        flagged_sessions = [
            session for session in SESSIONS_TO_PROCESS
            if any(date_str in session for date_str in UNCROPPED_DATE_STRINGS)
        ]
        if flagged_sessions:
            raise ValueError(
                f"""ERROR: CROPPED_VIDEOS=True but the following sessions appear to be uncropped longform videos:
{chr(10).join(f'  - {s}' for s in flagged_sessions)}
Check SESSIONS_TO_PROCESS or set CROPPED_VIDEOS=False."""
            )

    if not CROPPED_VIDEOS:
        CROPPED_DATE_STRINGS = ["2025_07", "2025_08"]  # If not using cropped video methods, don't use cropped videos. 
        flagged_sessions = [
            session for session in SESSIONS_TO_PROCESS
            if any(date_str in session for date_str in CROPPED_DATE_STRINGS)
        ]
        if flagged_sessions:
            raise ValueError(
                f"""ERROR: CROPPED_VIDEOS=False but the following sessions appear to be cropped videos:
{chr(10).join(f'  - {s}' for s in flagged_sessions)}
Check SESSIONS_TO_PROCESS or set CROPPED_VIDEOS=True."""
            )

    # Configuration summary
    print("\nConfiguration valid")
    print("\nGlobal Settings:")
    print(f"  Processing mode: {LONG_OR_SHORT.upper()}")
    print(f"  Anonymization: {'ENABLED' if ANONYMIZATION else 'DISABLED'}")
    print(f"  Background mode: {'CS on/off separation' if SEPARATE_CS_ON_OFF_BACKGROUNDS else 'Single averaged'}")
    print(f"  Resolution: {width} x {height}")
    print(f"  Frame rate: {FRAME_RATE} fps")

    print(f"\nBatch Information:")
    print(f"  Total sessions to process: {len(SESSIONS_TO_PROCESS)}")
    print(f"  Sessions:")
    for i, session in enumerate(SESSIONS_TO_PROCESS, 1):
        print(f"    {i}. {session}")

    if ANONYMIZATION:
        print(f"\nAnonymization files:")
        print(f"  Dictionary: {ANONYMIZATION_DICT_PATH}")
        print(f"  Lookup CSV: {LOOKUP_CSV_PATH}")
        print(f"  Experiment log: {EXPERIMENT_LOG}")

    print("\n" + "="*70)


# ============================================================================
# MAIN FUNCTION (Block 4 + Block 5)
# ============================================================================

def main():
    global ALL_TRIAL_DEFINITIONS

    # ========================================================================
    # Parse CLI arguments
    # ========================================================================
    parser = argparse.ArgumentParser(
        description="Batch process planarian video data: create JPEGs, MP4s, and background images."
    )
    parser.add_argument(
        '--long-or-short', choices=['long', 'short'], default=None,
        help="Override LONG_OR_SHORT flag (default: use value in script)"
    )
    parser.add_argument(
        '--anonymization', choices=['true', 'false'], default=None,
        help="Override ANONYMIZATION flag (default: use value in script)"
    )
    parser.add_argument(
        '--cropped-videos', choices=['true', 'false'], default=None,
        help="Override CROPPED_VIDEOS flag (default: use value in script)"
    )
    args = parser.parse_args()

    # Apply CLI overrides to module-level flags
    global LONG_OR_SHORT, ANONYMIZATION, CROPPED_VIDEOS
    if args.long_or_short is not None:
        LONG_OR_SHORT = args.long_or_short
    if args.anonymization is not None:
        ANONYMIZATION = args.anonymization.lower() == 'true'
    if args.cropped_videos is not None:
        CROPPED_VIDEOS = args.cropped_videos.lower() == 'true'

    # ========================================================================
    # Validate flags and initialize
    # ========================================================================
    validate_flags()
    initialize_anonymization()

    # ========================================================================
    # BATCH INITIALIZATION
    # ========================================================================

    print("="*70)
    print("STARTING BATCH PROCESSING")
    print("="*70)
    print(f"Total sessions: {len(SESSIONS_TO_PROCESS)}")
    print("="*70 + "\n")

    batch_start_time = time.time()

    batch_results = {
        'success': [],
        'failed': [],
        'skipped': [],
        'videos_created': 0,
        'backgrounds_created': 0
    }

    # ========================================================================
    # MAIN BATCH LOOP
    # ========================================================================

    for session_idx, session in enumerate(SESSIONS_TO_PROCESS):
        session_start_time = time.time()
        
        print("\n" + "="*70)
        print(f"SESSION {session_idx + 1}/{len(SESSIONS_TO_PROCESS)}: {session}")
        print("="*70)
        

        
        try:
            # ================================================================
            # STEP 1: Update session context
            # ================================================================
            
            SESSION_TYPE = update_session_context(session)
            print(f"  Type: {SESSION_TYPE}")
            
            # ================================================================
            # STEP 2: Calculate trial definitions
            # ================================================================

            if CROPPED_VIDEOS:  # Will use dictionaries if bin files have been postprocessed
                ALL_TRIAL_DEFINITIONS = calculate_trial_definitions_from_dictionary()
                
            else: # If no bin postprocessing (aka longform videos), use timestamps from video data
                ALL_TRIAL_DEFINITIONS = calculate_trial_definitions_from_timestamps()
            
            #ALL_TRIAL_DEFINITIONS = calculate_trial_definitions_from_timestamps()

            # This function will fail if the timestamps CSV can't be found. If the object is empty, check that the raw data is actually uploaded to the server.
            if ALL_TRIAL_DEFINITIONS is None:
                print(f"\tVariable ALL_TRIAL_DEFINITIONS is None.")
                find_missing_data(session) # This function will print the location of any folder or file matching the session ID. 
                
                raise RuntimeError("Failed to calculate trial definitions")
            
            # ================================================================
            # STEP 3: Get protocol trial labels (no filtering for TP)
            # ================================================================
            
            # Process ALL trials found in CSV
            TRIAL_DEFINITIONS = ALL_TRIAL_DEFINITIONS
            
            # Get protocol trial labels for anonymization
            if LONG_OR_SHORT == 'short' and SESSION_TYPE == 'TP':
                # For TP: Get protocol labels (e.g., [1, 3] for Block 7)
                # This validates that number of labels matches number of CSV trials
                PROTOCOL_TRIAL_LABELS = get_tp_protocol_trial_labels(len(TRIAL_DEFINITIONS))
            else:
                # For TC or long-form: Use sequential numbering
                PROTOCOL_TRIAL_LABELS = list(TRIAL_DEFINITIONS.keys())
            
            print(f"  Trials to process: {len(TRIAL_DEFINITIONS)}")
            
            if len(TRIAL_DEFINITIONS) == 0:
                raise RuntimeError("No trials found in stim_extra.csv")
            
            # ================================================================
            # STEP 4: Load coordinate regions file
            # ================================================================
            
            regions_pattern = os.path.join(COORD_FOLDER, "regions*")
            regions_files = glob.glob(regions_pattern)
            
            if regions_files:
                    print(f"Using regions file found at {regions_pattern}")
            
            if not regions_files:
                raise FileNotFoundError(f"No regions file found at {regions_pattern}")
            
            cropped_dims = np.load(regions_files[0])
            print(f"  Coordinate sets: {len(cropped_dims)}")
            
            # ================================================================
            # STEP 5: Load full video
            # ================================================================
            
            print("  Loading video data...")
            bin_pattern = os.path.join(HOLYLABS, 'Raw_data', binfile_to_segment, 'dat.bin')
            bin_files = glob.glob(bin_pattern)
            
            if not bin_files:
                raise FileNotFoundError(f"No .bin file found matching pattern: {bin_pattern}")
            
            SOURCE_VIDEO = bin_files[0]
            
            with open(SOURCE_VIDEO, 'rb') as f:
                data_ = np.fromfile(f, dtype=np.uint8)
                
                frame_size = width * height
                complete_frames = data_.shape[0] // frame_size
                data_ = data_[:complete_frames * frame_size]
                data_ = data_.reshape(complete_frames, height, width)
            
            print(f"  Video loaded: {data_.shape[0]} frames")
            
            # Validate trial ranges if doing short-form processing
            if LONG_OR_SHORT == 'short':
                validate_trial_ranges(data_.shape, TRIAL_DEFINITIONS)
            
            # ================================================================
            # STEP 6: Process videos (long-form or short-form)
            # ================================================================
            
            print(f"\n  Processing mode: {LONG_OR_SHORT.upper()}")
            session_video_count = 0
            
            if LONG_OR_SHORT == 'long':
                # ============================================================
                # LONG-FORM PROCESSING: No temporal segmentation
                # ============================================================
                
                for coord_idx, coords in enumerate(cropped_dims):
                    print(f"\n    Coordinate set {coord_idx + 1}/{len(cropped_dims)}") # This is where it's hanging
                    
                    # Create descriptive folder name
                    coord_string = f"{coords[0]}_{coords[1]}_{coords[2]}_{coords[3]}"
                    folder_name = f"{binfile_to_segment}_regions_{coord_string}_fullvideo"
                    
                    # Create output directory
                    SOURCE_FRAMES_DLC = Path(os.path.join(
                        DLC_PROJECT_PATH, 'unlabeled-data', binfile_to_segment, folder_name
                    ))
                    os.makedirs(SOURCE_FRAMES_DLC, exist_ok=True)
                    
                    # Calculate crop boundaries
                    x_start = int(min(coords[0], coords[2]))
                    x_end = int(max(coords[0], coords[2]))
                    y_start = int(min(coords[1], coords[3]))
                    y_end = int(max(coords[1], coords[3]))
                    
                    x_start = max(0, x_start)
                    y_start = max(0, y_start)
                    x_end = min(data_.shape[2], x_end)
                    y_end = min(data_.shape[1], y_end)
                    
                    # Process ALL frames
                    for frame_idx in range(data_.shape[0]):
                        frame = data_[frame_idx]
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        cropped = frame_bgr[y_start:y_end, x_start:x_end]
                        
                        jpeg_filename = f"{frame_idx:05d}.jpeg"
                        jpeg_path = os.path.join(SOURCE_FRAMES_DLC, jpeg_filename)
                        cv2.imwrite(jpeg_path, cropped)
                        
                        # Overwriting progress bar
                        if (frame_idx + 1) % 100 == 0 or frame_idx == data_.shape[0] - 1:
                            # If not 1799 frames to be made, cancel - .bin video is probably messed up
                            if (data_.shape[0] not in [1796, 1797, 1798, 1799, 1800]) and (ANONYMIZATION is False):
                                data_shape_error = (f"Invalid data shape: Expected 1798, 1799, or 1800, but got {data_.shape[0]}")
                                print(data_shape_error)
                                raise ValueError(f"Invalid data shape: Expected 1798, 1799, or 1800, but got {data_.shape[0]}")
                            progress = ((frame_idx + 1) / data_.shape[0]) * 100
                            print(f"\r      Saving JPEGs: {progress:.0f}% [{frame_idx+1}/{data_.shape[0]}]", end='', flush=True)
                    
                    print()  # New line after progress bar
                    
                    # Create MP4 video
                    make_DLC_videos_from_jpegs(folder_name, SOURCE_FRAMES_DLC)
                    session_video_count += 1
            
            else:
                # ============================================================
                # SHORT-FORM PROCESSING: Temporal segmentation
                # ============================================================
                
                for coord_idx, coords in enumerate(cropped_dims):
                    print(f"\n    Coordinate set {coord_idx + 1}/{len(cropped_dims)}")
                    
                    # Calculate crop boundaries
                    x_start = int(min(coords[0], coords[2]))
                    x_end = int(max(coords[0], coords[2]))
                    y_start = int(min(coords[1], coords[3]))
                    y_end = int(max(coords[1], coords[3]))
                    
                    x_start = max(0, x_start)
                    y_start = max(0, y_start)
                    x_end = min(data_.shape[2], x_end)
                    y_end = min(data_.shape[1], y_end)
                    
                    # Process each trial separately
                    for csv_trial_idx, (trial_num, (start_frame, end_frame)) in enumerate(TRIAL_DEFINITIONS.items()):
                        # Get the protocol trial number for labeling in lookup CSV
                        protocol_trial_num = PROTOCOL_TRIAL_LABELS[csv_trial_idx]
                        
                        # Display both CSV trial number and protocol label
                        if SESSION_TYPE == 'TP' and protocol_trial_num != trial_num:
                            print(f"      CSV trial {trial_num} (protocol trial {protocol_trial_num}): frames {start_frame}-{end_frame}")
                        else:
                            print(f"      Trial {trial_num}: frames {start_frame}-{end_frame}")
                        
                        # Generate folder name
                        if ANONYMIZATION:
                            anonymous_name = generate_anonymous_video_name(
                                adjectives, nouns, used_names, LOOKUP_CSV_PATH
                            )
                            folder_name = anonymous_name
                            print(f"        Anonymous name: {anonymous_name}")
                        else:
                            coord_string = f"{coords[0]}_{coords[1]}_{coords[2]}_{coords[3]}"
                            folder_name = f"{binfile_to_segment}_coords_{coord_string}_Trial{protocol_trial_num}"
                        
                        # Extract frames for this trial
                        trial_frames = data_[start_frame:end_frame+1]
                        
                        # Create output folder
                        SOURCE_FRAMES_DLC = Path(os.path.join(
                            DLC_PROJECT_PATH, 'unlabeled-data', binfile_to_segment, folder_name
                        ))
                        os.makedirs(SOURCE_FRAMES_DLC, exist_ok=True)
                        
                        # Save JPEGs
                        for frame_idx, frame in enumerate(trial_frames):
                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                            cropped = frame_bgr[y_start:y_end, x_start:x_end]
                            
                            original_frame_num = start_frame + frame_idx
                            jpeg_filename = f"{original_frame_num:05d}.jpeg"
                            jpeg_path = os.path.join(SOURCE_FRAMES_DLC, jpeg_filename)
                            cv2.imwrite(jpeg_path, cropped)
                            
                        # Confirm that expected 1799 frames have been made
                        if not ANONYMIZATION:  # Not necessary for short videos
                            print("I need something here to run cell")
                            # look at jpeg_path and jpeg_filename
                            # count contents of folder
                            # Does it equal 1799? T/F print a nice statement depending
                        
                        # Update lookup CSV with PROTOCOL trial number
                        if ANONYMIZATION:
                            update_lookup_csv_with_trial(
                                LOOKUP_CSV_PATH, 
                                anonymous_name, 
                                binfile_to_segment, 
                                coords, 
                                protocol_trial_num,  # Use protocol label, not CSV trial number
                                start_frame, 
                                end_frame
                            )
                        
                        # Create MP4 video
                        make_DLC_videos_from_jpegs(folder_name, SOURCE_FRAMES_DLC)
                        session_video_count += 1
                        
                            
                                    
            # ================================================================
            # STEP 7: Create background images
            # ================================================================
            
            print(f"\n  Creating background images...")
            create_backgrounds()
            
            if SEPARATE_CS_ON_OFF_BACKGROUNDS:
                session_background_count = len(cropped_dims) * 2  # CSon and CSoff
            else:
                session_background_count = len(cropped_dims)  # Single averaged
            
            # ================================================================
            # STEP 8: Clean up memory
            # ================================================================
            
            del data_
            gc.collect()
            print("This part worked [3]")
            
            # ================================================================
            # SESSION COMPLETE
            # ================================================================
            
            session_duration = time.time() - session_start_time
            
            print(f"\n  Session complete: {session_duration/60:.1f} minutes")
            print(f"    Videos created: {session_video_count}")
            print(f"    Backgrounds created: {session_background_count}")
            
            batch_results['success'].append(session)
            batch_results['videos_created'] += session_video_count
            batch_results['backgrounds_created'] += session_background_count
            
            # Progress tracking
            elapsed_total = time.time() - batch_start_time
            avg_time_per_session = elapsed_total / (session_idx + 1)
            remaining_sessions = len(SESSIONS_TO_PROCESS) - (session_idx + 1)
            estimated_remaining = avg_time_per_session * remaining_sessions
            
            print(f"\n  Batch progress: {session_idx + 1}/{len(SESSIONS_TO_PROCESS)} sessions complete")
            print(f"  Estimated time remaining: {estimated_remaining/60:.1f} minutes")
            
        except FileNotFoundError as e:
            print(f"\n  ERROR: Missing required files")
            print(f"    {e}")
            batch_results['skipped'].append((session, str(e)))
            
        except Exception as e:
        #     print("This is the one that keeps popping up.")
            
            print(f"\n  ERROR: Processing failed")
            print(f"    {e}")
            batch_results['failed'].append((session, str(e)))
            
        finally:
            # Always clean up memory, even if there was an error
            if 'data_' in locals():
                del data_
            gc.collect()

    print("\n" + "="*70)
    print("BATCH PROCESSING COMPLETE")
    print("="*70)

    # ========================================================================
    # SUMMARY REPORT (Block 5)
    # ========================================================================

    print("\n" + "="*70)
    print("BATCH PROCESSING SUMMARY REPORT")
    print("="*70)

    batch_total_time = time.time() - batch_start_time

    # Timing summary
    print("\nTiming:")
    print(f"  Total elapsed time: {batch_total_time/60:.1f} minutes ({batch_total_time/3600:.2f} hours)")
    print(f"  Average time per session: {(batch_total_time/len(SESSIONS_TO_PROCESS))/60:.1f} minutes")

    # Session summary
    total_sessions = len(SESSIONS_TO_PROCESS)
    successful_sessions = len(batch_results['success'])
    failed_sessions = len(batch_results['failed'])
    skipped_sessions = len(batch_results['skipped'])

    print("\nSessions:")
    print(f"  Total sessions: {total_sessions}")
    print(f"  Successful: {successful_sessions}")
    print(f"  Failed: {failed_sessions}")
    print(f"  Skipped: {skipped_sessions}")

    # Output summary
    print("\nOutputs created:")
    print(f"  Videos: {batch_results['videos_created']}")
    print(f"  Background images: {batch_results['backgrounds_created']}")

    # Successful sessions
    if batch_results['success']:
        print("\nSuccessful sessions:")
        for session in batch_results['success']:
            print(f"  - {session}")

    # Skipped sessions
    if batch_results['skipped']:
        print("\nSkipped sessions (missing files):")
        for session, error in batch_results['skipped']:
            print(f"  - {session}")
            print(f"      Reason: {error}")

    # Failed sessions
    if batch_results['failed']:
        print("\nFailed sessions (errors during processing):")
        for session, error in batch_results['failed']:
            print(f"  - {session}")
            print(f"      Error: {error}")

    # Output locations
    print("\nOutput locations:")
    print(f"  Videos: {DLC_PROJECT_PATH}/videos/")
    print(f"  JPEGs: {DLC_PROJECT_PATH}/unlabeled-data/")
    print(f"  Backgrounds: {HOLYLABS}/Calibration_images/Full_videos/")

    if ANONYMIZATION:
        print(f"  Lookup CSV: {LOOKUP_CSV_PATH}")

    # Final status
    print("\n" + "="*70)

    if failed_sessions == 0 and skipped_sessions == 0:
        print("BATCH PROCESSING COMPLETED SUCCESSFULLY")
        print("All sessions processed without errors.")
    elif failed_sessions > 0:
        print("BATCH PROCESSING COMPLETED WITH ERRORS")
        print(f"{failed_sessions} session(s) failed - see error log above.")
    else:
        print("BATCH PROCESSING COMPLETED WITH WARNINGS")
        print(f"{skipped_sessions} session(s) skipped - see log above.")

    print("="*70)


if __name__ == '__main__':
    main()
