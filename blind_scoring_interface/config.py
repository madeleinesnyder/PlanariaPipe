import os

# Video folder paths - modify these as needed
#HOME = ""
HOME = "/Users/zacharykelso/Desktop/Blind_Scoring_Interface/"
# HOME = os.path.dirname(os.path.abspath(__file__))

UNANNOTATED_VIDEOS_PATH = f"{HOME}/Unannotated_videos"
ANNOTATED_VIDEOS_PATH = f"{HOME}/Annotated_videos"

# Supported video file extensions
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']

# Profile options for the dropdown - modify these as needed
PROFILES = ["ZK"]

# Keyboard shortcuts - modify these as needed
KEYBINDINGS = {
    'play_pause': '<space>',
    'frame_forward': '<Right>',
    'frame_backward': '<Left>',
    'increment_turns': '<Key-1>',
    'decrement_turns': '<Shift-Key-1>',
    'increment_contractions': '<Key-3>',
    'decrement_contractions': '<Shift-Key-3>',
    'save': '<Control-s>',
    'save_confirm_yes': '<Key-y>',
    'save_confirm_no': '<Key-n>'
}

# Create directories if they don't exist
os.makedirs(UNANNOTATED_VIDEOS_PATH, exist_ok=True)
os.makedirs(ANNOTATED_VIDEOS_PATH, exist_ok=True)

# CSV save location - modify as needed
CSV_SAVE_PATH = f"{HOME}/Data"

# Help text for the Instructions button - UPDATED
HELP_TEXT = """
VIDEO ANNOTATION INTERFACE - USER GUIDE

SUPPORTED MEDIA:
Video files are loaded as a bunch of serially named JPEGs (001, 002, 003, ...) in folders containing randomized nonsense names. Each folder of JPEGs represents one video.
• Video files: .mp4, .avi, .mov, .mkv, .wmv, .flv, .webm
• JPEG sequences: Folders containing numbered JPEG files (0001.jpg, 0002.jpg, etc.)
• JPEG folders appear with "/" suffix in the file browser

KEYBOARD SHORTCUTS:
These are the buttons you can press to control what happens to the video.
• Spacebar: Play/Pause video
• Left/Right arrows: Move one frame forward or backward
• Press 1 to increment turns, Shift+1 to decrement turns
• Press 3 to increment contractions, Shift+3 to decrement contractions
• Ctrl+S: Open save dialog
• Y/N: Confirm/Cancel save (when dialog is open)

SCORING:
• Click counter buttons or use keyboard shortcuts to mark behaviors on the frame that they start happening
• Each frame can have max 1 turn mark and 1 contraction mark -- the video is 10 frames per second, so >1 turn/con on one frame is basically impossible
• Cyan bars on timeline = Turn timestamps
• Orange bars on timeline = Contraction timestamps
• Use interactive slider to jump to specific frames

VIDEO DISPLAY:
• "Video Display Options" button allows brightness/contrast adjustment
• These adjustments only affect display, not the source files -- they're just for the user to see better
• Use Reset button in display options to return to normal

NOTES:
• Type notes in the notes box (keyboard shortcuts disabled while typing)
• Notes are saved with your annotations
• Reset button clears notes along with counters
• Press escape key to to exit notetaking and access normal keyboard controls
• Too many notes is better than too few! Write down whatever is even slightly interesting

SAVING:
The saving works by saving your timestamps and associated video name to a CSV file. You can save a new CSV every time or append
to an old CSV. You can also choose to save to a copy of an old CSV. I recommend making a copy every now and then just to make sure
you don't accidentally write over and old data. It is better to have too many copies than to lose data on accident.
• Select profile before saving
• Choose to save to new file or append to existing CSV
• ↻ button refreshes CSV file list
• The "Expand" button just makes the saving section larger so that it is easier to read.
• Check "Create copy" to backup existing files when appending
• Red warning appears if video already annotated in selected CSV

NAVIGATION:
• Load videos/JPEG folders from left panel
• Move files between annotated/unannotated folders (Saving automatically moves a video from Unann. to Ann.)
• Use video controls and slider to navigate through frames

For technical support, contact Zachary Kelso (zkelso@fas.harvard.edu)
"""

# Create CSV directory if it doesn't exist
os.makedirs(CSV_SAVE_PATH, exist_ok=True)