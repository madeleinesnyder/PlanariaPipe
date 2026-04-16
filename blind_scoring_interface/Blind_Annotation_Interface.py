import tkinter as tk
from tkinter import ttk, messagebox
import os
import cv2
import shutil
from PIL import Image, ImageTk, ImageEnhance
import numpy as np
import config
import csv
from datetime import datetime

class VideoAnnotationInterface:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Video Annotation Interface")
        
        # Video player variables
        self.video_capture = None
        self.current_video_path = None
        self.current_frame_number = 0
        self.total_frames = 0
        self.is_playing = False
        self.current_frame_image = None
        self.selected_video_info = None

        # Store original filenames for mapping display names to actual files
        self.unannotated_original_names = []
        self.annotated_original_names = []
        
        # JPEG sequence variables
        self.is_jpeg_sequence = False
        self.jpeg_file_list = []
        
        # Display adjustment variables
        self.brightness_adjustment = 0  # -100 to 100
        self.contrast_adjustment = 0    # -100 to 100
        
        # Timestamp tracking variables
        self.turn_timestamps = []  # List of frame numbers with turn marks
        self.contraction_timestamps = []  # List of frame numbers with contraction marks
        
        self.root.geometry("1200x900")
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after_idle(self.root.attributes, '-topmost', False)
        
        # Bind keyboard shortcuts
        self.setup_keyboard_bindings()
        
        self.setup_layout()
        self.load_video_lists()
        
    def setup_keyboard_bindings(self):
        """Setup keyboard shortcuts"""
        self.root.bind('<Key>', self.on_key_press)
        self.root.focus_set()
        
    def on_notes_focus_in(self, event):
        """Disable global keyboard shortcuts when typing in notes"""
        self.root.unbind('<Key>')

    def on_notes_focus_out(self, event):
        """Re-enable global keyboard shortcuts when leaving notes"""
        self.root.bind('<Key>', self.on_key_press)
        
    def on_notes_escape(self, event):
        """Remove focus from notes box when Escape is pressed"""
        self.root.focus_set()  # Give focus back to main window
        return 'break'  # Prevent default Escape behavior
        
    def on_key_press(self, event):
        """Handle keyboard shortcuts"""
        key = event.keysym
        char = event.char
        state = event.state
        
        ctrl_pressed = (state & 0x0004) != 0
        
        # Handle shifted number keys by their produced characters
        if char == '!':  # Shift+1 produces '!'
            self.decrement_counter('turns')
            return
        elif char == '#':  # Shift+3 produces '#'
            self.decrement_counter('contractions')
            return
        elif key == '1':
            self.increment_counter('turns')
            return
        elif key == '3':
            self.increment_counter('contractions')
            return
        
        # Other shortcuts
        if key == 'space':
            self.toggle_play_pause()
        elif key == 'Right':
            self.frame_forward()
        elif key == 'Left':
            self.frame_backward()
        elif key == 's' and ctrl_pressed:
            self.save_data()
            
    def extract_video_title(self, filename):
        """Extract display title from video filename or folder name"""
        # Remove trailing slash if present (for JPEG folders)
        if filename.endswith('/'):
            filename = filename[:-1]
            
        if filename.startswith("Video_"):
            # Remove "Video_" prefix and file extension
            if filename.endswith(".mp4"):
                title_part = filename[6:-4]  # Remove "Video_" and ".mp4"
            else:
                title_part = filename[6:]    # Remove "Video_" only (for folders)
            return title_part
        return filename
        
    def show_help(self):
        """Display help dialog"""
        help_dialog = tk.Toplevel(self.root)
        help_dialog.title("Instructions - Video Annotation Interface")
        help_dialog.geometry("600x500")
        help_dialog.transient(self.root)
        help_dialog.grab_set()
        
        # Center the dialog
        help_dialog.geometry("+%d+%d" % (self.root.winfo_rootx() + 100, self.root.winfo_rooty() + 50))
        
        # Create scrollable text widget
        text_frame = tk.Frame(help_dialog)
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(text_frame, wrap='word', font=('Arial', 10))
        scrollbar = tk.Scrollbar(text_frame, orient='vertical', command=text_widget.yview)
        text_widget.config(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Insert help text
        text_widget.insert('1.0', config.HELP_TEXT)
        text_widget.config(state='disabled')  # Make read-only
        
        # Close button
        tk.Button(help_dialog, text='Close', command=help_dialog.destroy,
                 font=('Arial', 11), width=10).pack(pady=10)

    def show_display_options(self):
        """Show video display adjustment dialog"""
        display_dialog = tk.Toplevel(self.root)
        display_dialog.title("Video Display Options")
        display_dialog.geometry("400x200")
        display_dialog.transient(self.root)
        display_dialog.grab_set()
        
        # Center the dialog
        display_dialog.geometry("+%d+%d" % (self.root.winfo_rootx() + 100, self.root.winfo_rooty() + 100))
        
        # Brightness control
        brightness_frame = tk.Frame(display_dialog)
        brightness_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(brightness_frame, text="Brightness:", font=('Arial', 11)).pack(side='left')
        self.brightness_var = tk.DoubleVar(value=self.brightness_adjustment)
        brightness_scale = tk.Scale(brightness_frame, from_=-100, to=100, orient='horizontal',
                                   variable=self.brightness_var, command=self.update_display_adjustments)
        brightness_scale.pack(side='right', fill='x', expand=True, padx=(10, 0))
        
        # Contrast control
        contrast_frame = tk.Frame(display_dialog)
        contrast_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(contrast_frame, text="Contrast:", font=('Arial', 11)).pack(side='left')
        self.contrast_var = tk.DoubleVar(value=self.contrast_adjustment)
        contrast_scale = tk.Scale(contrast_frame, from_=-100, to=100, orient='horizontal',
                                 variable=self.contrast_var, command=self.update_display_adjustments)
        contrast_scale.pack(side='right', fill='x', expand=True, padx=(10, 0))
        
        # Buttons
        button_frame = tk.Frame(display_dialog)
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text='Reset', command=self.reset_display_adjustments,
                 font=('Arial', 10), width=10).pack(side='left', padx=5)
        tk.Button(button_frame, text='Close', command=display_dialog.destroy,
                 font=('Arial', 10), width=10).pack(side='left', padx=5)

    def update_display_adjustments(self, value=None):
        """Update display adjustments and refresh frame"""
        self.brightness_adjustment = self.brightness_var.get()
        self.contrast_adjustment = self.contrast_var.get()
        self.update_frame_display()

    def reset_display_adjustments(self):
        """Reset display adjustments to defaults"""
        self.brightness_var.set(0)
        self.contrast_var.set(0)
        self.brightness_adjustment = 0
        self.contrast_adjustment = 0
        self.update_frame_display()
                 
    def get_csv_files(self):
        """Get list of existing CSV files in save directory"""
        if not os.path.exists(config.CSV_SAVE_PATH):
            return []
        
        csv_files = []
        try:
            for file in os.listdir(config.CSV_SAVE_PATH):
                if file.lower().endswith('.csv'):
                    csv_files.append(file)
        except:
            pass
        return sorted(csv_files)

    def check_video_already_annotated(self, csv_filename, video_filename):
        """Check if video is already annotated in the selected CSV file"""
        if csv_filename == "New file":
            return False
            
        csv_path = os.path.join(config.CSV_SAVE_PATH, csv_filename)
        if not os.path.exists(csv_path):
            return False
            
        try:
            with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                next(reader, None)  # Skip header
                for row in reader:
                    if row and row[0] == video_filename:
                        return True
        except:
            pass
        return False
        
    def generate_csv_filename(self):
        """Generate new CSV filename based on profile and timestamp"""
        now = datetime.now()
        profile_name = self.profile_var.get().replace(" ", "_")
        timestamp = now.strftime("%Y%m%d_%H%M")
        return f"Blind_Planarian_Scoring_{profile_name}_{timestamp}.csv"
    
    def show_expanded_save_options(self):
        """Show expanded save options dialog with larger, more readable interface"""
        expanded_dialog = tk.Toplevel(self.root)
        expanded_dialog.title("Save Options - Expanded View")
        expanded_dialog.geometry("500x450")
        expanded_dialog.transient(self.root)
        expanded_dialog.grab_set()
        
        # Center the dialog
        expanded_dialog.geometry("+%d+%d" % (self.root.winfo_rootx() + 100, self.root.winfo_rooty() + 100))
        
        main_frame = tk.Frame(expanded_dialog, padx=20, pady=20)
        main_frame.pack(fill='both', expand=True)
        
        # Profile section
        profile_frame = tk.LabelFrame(main_frame, text="Profile Selection", font=('Arial', 11, 'bold'), padx=10, pady=10)
        profile_frame.pack(fill='x', pady=(0, 15))
        
        tk.Label(profile_frame, text='Select Profile:', font=('Arial', 10)).pack(anchor='w', pady=(0, 5))
        
        expanded_profile_var = tk.StringVar(value=self.profile_var.get())
        profile_dropdown = ttk.Combobox(profile_frame, textvariable=expanded_profile_var,
                                       values=getattr(config, 'PROFILES', ["Default"]),
                                       state='readonly', width=30, font=('Arial', 11))
        profile_dropdown.pack(fill='x')
        
        # CSV file selection section
        csv_frame = tk.LabelFrame(main_frame, text="CSV File Selection", font=('Arial', 11, 'bold'), padx=10, pady=10)
        csv_frame.pack(fill='both', expand=True, pady=(0, 15))
        
        tk.Label(csv_frame, text='Select or create CSV file:', font=('Arial', 10)).pack(anchor='w', pady=(0, 5))
        
        # Listbox for CSV files (much larger and more readable)
        listbox_frame = tk.Frame(csv_frame)
        listbox_frame.pack(fill='both', expand=True)
        
        csv_listbox = tk.Listbox(listbox_frame, height=10, selectmode='single', font=('Arial', 11))
        scrollbar = tk.Scrollbar(listbox_frame, orient='vertical', command=csv_listbox.yview)
        csv_listbox.config(yscrollcommand=scrollbar.set)
        
        csv_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Populate listbox
        csv_files = ["New file"] + self.get_csv_files()
        for csv_file in csv_files:
            csv_listbox.insert(tk.END, csv_file)
        
        # Select current file
        current_csv = self.csv_file_var.get()
        if current_csv in csv_files:
            csv_listbox.selection_set(csv_files.index(current_csv))
            csv_listbox.see(csv_files.index(current_csv))
        
        # Create copy checkbox
        expanded_copy_var = tk.BooleanVar(value=self.create_copy_var.get())
        copy_check = tk.Checkbutton(csv_frame, text='Create new copy of selected file', 
                                    variable=expanded_copy_var, font=('Arial', 10))
        copy_check.pack(anchor='w', pady=(10, 0))
        
        # Preview section
        preview_frame = tk.LabelFrame(main_frame, text="Current Selection", font=('Arial', 10, 'bold'), padx=10, pady=10)
        preview_frame.pack(fill='x', pady=(0, 15))
        
        preview_label = tk.Label(preview_frame, text=f"Profile: {self.profile_var.get()}\nCSV: {self.csv_file_var.get()}", 
                                font=('Arial', 10), justify='left', anchor='w')
        preview_label.pack(fill='x')
        
        # Buttons
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=(10, 0))
        
        def apply_and_close():
            """Apply selections and close dialog"""
            # Update profile
            self.profile_var.set(expanded_profile_var.get())
            
            # Update CSV file selection
            selection = csv_listbox.curselection()
            if selection:
                selected_csv = csv_listbox.get(selection[0])
                self.csv_file_var.set(selected_csv)
            
            # Update create copy option
            self.create_copy_var.set(expanded_copy_var.get())
            
            expanded_dialog.destroy()
        
        tk.Button(button_frame, text='Apply', command=apply_and_close,
                 font=('Arial', 11, 'bold'), bg='lightgreen', width=12).pack(side='left', padx=5)
        tk.Button(button_frame, text='Close', command=expanded_dialog.destroy,
                 font=('Arial', 11), width=12).pack(side='left', padx=5)
        
    def setup_layout(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        main_frame.grid_columnconfigure(0, weight=0, minsize=350)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        
        # === LEFT PANEL ===
        self.create_file_browser_panel(main_frame)
        
        # === RIGHT PANEL ===
        right_frame = tk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky='nsew', padx=(5,0))
        
        right_frame.grid_rowconfigure(0, weight=1)  # Video section gets most space
        right_frame.grid_rowconfigure(1, weight=0, minsize=200)  # Compact scoring section
        right_frame.grid_columnconfigure(0, weight=1)
        
        self.create_video_panel(right_frame)
        self.create_scoring_panel(right_frame)

    def create_file_browser_panel(self, parent):
        """Left panel with working listboxes"""
        container = tk.Frame(parent, width=350, relief='sunken', borderwidth=2)
        container.grid(row=0, column=0, sticky='nsew', padx=(0,5))
        container.grid_propagate(False)
        container.grid_rowconfigure(0, weight=0)  # Header
        container.grid_rowconfigure(1, weight=1)  # Top box
        container.grid_rowconfigure(2, weight=0)  # Header
        container.grid_rowconfigure(3, weight=1)  # Bottom box
        container.grid_rowconfigure(4, weight=0)  # Buttons
        container.grid_columnconfigure(0, weight=1)
        
        # === TOP BOX HEADER ===
        self.unannotated_header = tk.Label(container, 
                                          text='Unannotated items (loading...)',
                                          font=('Arial', 11, 'bold'),
                                          relief='raised', borderwidth=1)
        self.unannotated_header.grid(row=0, column=0, sticky='ew', pady=5, padx=5)
        
        # === TOP LISTBOX ===
        unannotated_frame = tk.Frame(container)
        unannotated_frame.grid(row=1, column=0, sticky='nsew', pady=(0,10), padx=10)
        
        unannotated_frame.grid_rowconfigure(0, weight=1)
        unannotated_frame.grid_columnconfigure(0, weight=1)
        
        self.unannotated_listbox = tk.Listbox(unannotated_frame, 
                                             height=8, 
                                             selectmode='single',
                                             font=('Arial', 9))
        scrollbar1 = tk.Scrollbar(unannotated_frame, orient='vertical', 
                                 command=self.unannotated_listbox.yview)
        self.unannotated_listbox.config(yscrollcommand=scrollbar1.set)
        
        self.unannotated_listbox.grid(row=0, column=0, sticky='nsew')
        scrollbar1.grid(row=0, column=1, sticky='ns')
        
        # Bind selection
        self.unannotated_listbox.bind('<<ListboxSelect>>', 
                                     lambda e: self.on_listbox_select('unannotated'))
        
        # === BOTTOM BOX HEADER ===
        self.annotated_header = tk.Label(container, 
                                        text='Annotated items (loading...)',
                                        font=('Arial', 11, 'bold'),
                                        relief='raised', borderwidth=1)
        self.annotated_header.grid(row=2, column=0, sticky='ew', pady=5, padx=5)
        
        # === BOTTOM LISTBOX ===
        annotated_frame = tk.Frame(container)
        annotated_frame.grid(row=3, column=0, sticky='nsew', pady=(0,10), padx=10)
        
        annotated_frame.grid_rowconfigure(0, weight=1)
        annotated_frame.grid_columnconfigure(0, weight=1)
        
        self.annotated_listbox = tk.Listbox(annotated_frame, 
                                           height=8, 
                                           selectmode='single',
                                           font=('Arial', 9))
        scrollbar2 = tk.Scrollbar(annotated_frame, orient='vertical', 
                                 command=self.annotated_listbox.yview)
        self.annotated_listbox.config(yscrollcommand=scrollbar2.set)
        
        self.annotated_listbox.grid(row=0, column=0, sticky='nsew')
        scrollbar2.grid(row=0, column=1, sticky='ns')
        
        # Bind selection
        self.annotated_listbox.bind('<<ListboxSelect>>', 
                                   lambda e: self.on_listbox_select('annotated'))
        
        # === CONTROL BUTTONS ===
        button_frame = tk.Frame(container)
        button_frame.grid(row=4, column=0, pady=10, padx=10, sticky='ew')
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)
        
        load_btn = tk.Button(button_frame, text='Load Video', 
                            command=self.load_selected_video,
                            font=('Arial', 9, 'bold'))
        load_btn.grid(row=0, column=0, padx=(0,2), pady=2, sticky='ew')
        
        move_btn = tk.Button(button_frame, text='Move File', 
                            command=self.move_selected_file,
                            font=('Arial', 9, 'bold'))
        move_btn.grid(row=0, column=1, padx=(2,0), pady=2, sticky='ew')
        
        refresh_btn = tk.Button(button_frame, text='Refresh Lists', 
                               command=self.load_video_lists,
                               font=('Arial', 9))
        refresh_btn.grid(row=1, column=0, columnspan=2, pady=(5,0), sticky='ew')
        
        # Selection indicator
        self.selection_label = tk.Label(container, text='No video selected', 
                                       font=('Arial', 9, 'bold'), relief='sunken', borderwidth=1)
        self.selection_label.grid(row=5, column=0, sticky='ew', padx=5, pady=5)
        
    def get_video_files(self, directory):
        """Get list of video files from directory"""
        if not os.path.exists(directory):
            return []
        
        video_files = []
        try:
            for file in os.listdir(directory):
                if any(file.lower().endswith(ext) for ext in config.VIDEO_EXTENSIONS):
                    video_files.append(file)
        except:
            pass
        return sorted(video_files)

    def get_jpeg_folders(self, directory):
        """Get list of folders containing JPEG sequences"""
        if not os.path.exists(directory):
            return []
        
        jpeg_folders = []
        try:
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isdir(item_path):
                    # Check if folder contains JPEG files
                    jpeg_files = [f for f in os.listdir(item_path) 
                                 if f.lower().endswith(('.jpg', '.jpeg'))]
                    if jpeg_files:
                        jpeg_folders.append(item)
        except:
            pass
        return sorted(jpeg_folders)
    
    def load_video_lists(self):
        """Load video files and JPEG folders into the listboxes"""
        print("Loading video lists...")
    
        unannotated_videos = self.get_video_files(config.UNANNOTATED_VIDEOS_PATH)
        unannotated_jpegs = self.get_jpeg_folders(config.UNANNOTATED_VIDEOS_PATH)
        unannotated_all = unannotated_videos + [f + "/" for f in unannotated_jpegs]
    
        annotated_videos = self.get_video_files(config.ANNOTATED_VIDEOS_PATH)
        annotated_jpegs = self.get_jpeg_folders(config.ANNOTATED_VIDEOS_PATH)
        annotated_all = annotated_videos + [f + "/" for f in annotated_jpegs]
    
        print(f"Found {len(unannotated_all)} unannotated items ({len(unannotated_videos)} videos, {len(unannotated_jpegs)} JPEG folders)")
        print(f"Found {len(annotated_all)} annotated items ({len(annotated_videos)} videos, {len(annotated_jpegs)} JPEG folders)")
    
        # Store original names and display cleaned names
        self.unannotated_original_names = sorted(unannotated_all)
        self.annotated_original_names = sorted(annotated_all)
    
        # Clear and populate unannotated
        self.unannotated_listbox.delete(0, tk.END)
        for item in self.unannotated_original_names:
            display_name = self.extract_video_title(item)
            self.unannotated_listbox.insert(tk.END, display_name)
    
        # Clear and populate annotated
        self.annotated_listbox.delete(0, tk.END)
        for item in self.annotated_original_names:
            display_name = self.extract_video_title(item)
            self.annotated_listbox.insert(tk.END, display_name)
        
        # Update headers
        self.unannotated_header.config(text=f'Unannotated items ({len(unannotated_all)})')
        self.annotated_header.config(text=f'Annotated items ({len(annotated_all)})')
    
        print("Video/JPEG lists loaded!")
        
    def on_listbox_select(self, list_type):
        """Handle listbox selection for videos and JPEG folders"""
        if list_type == 'unannotated':
            selection = self.unannotated_listbox.curselection()
            if selection:
                index = selection[0]
                # Get original filename using index
                item_name = self.unannotated_original_names[index]
                
                # Check if it's a JPEG folder (ends with /)
                if item_name.endswith('/'):
                    folder_name = item_name[:-1]  # Remove trailing slash
                    item_path = os.path.join(config.UNANNOTATED_VIDEOS_PATH, folder_name)
                else:
                    item_path = os.path.join(config.UNANNOTATED_VIDEOS_PATH, item_name)
                
                # Get display name for label
                display_name = self.extract_video_title(item_name)
                self.selected_video_info = (item_path, item_name, 'unannotated')
                self.selection_label.config(text=f"Selected: {display_name}")
                # Clear other selection
                self.annotated_listbox.selection_clear(0, tk.END)
        else:
            selection = self.annotated_listbox.curselection()
            if selection:
                index = selection[0]
                # Get original filename using index
                item_name = self.annotated_original_names[index]
                
                # Check if it's a JPEG folder (ends with /)
                if item_name.endswith('/'):
                    folder_name = item_name[:-1]  # Remove trailing slash
                    item_path = os.path.join(config.ANNOTATED_VIDEOS_PATH, folder_name)
                else:
                    item_path = os.path.join(config.ANNOTATED_VIDEOS_PATH, item_name)
                    
                # Get display name for label
                display_name = self.extract_video_title(item_name)
                self.selected_video_info = (item_path, item_name, 'annotated')
                self.selection_label.config(text=f"Selected: {display_name}")
                # Clear other selection
                self.unannotated_listbox.selection_clear(0, tk.END)
    

    def load_selected_video(self):
        """Load the selected video into the player"""
        if not self.selected_video_info:
            messagebox.showwarning("No Selection", "Please select a video to load.")
            return
            
        video_path, video_name, source = self.selected_video_info
        self.load_video(video_path)
        
    def move_selected_file(self):
        """Move selected file between annotated/unannotated folders"""
        if not self.selected_video_info:
            messagebox.showwarning("No Selection", "Please select a video to move.")
            return
            
        video_path, video_name, source = self.selected_video_info
        
        try:
            if source == 'unannotated':
                if video_name.endswith('/'):
                    folder_name = video_name[:-1]
                    destination = os.path.join(config.ANNOTATED_VIDEOS_PATH, folder_name)
                else:
                    destination = os.path.join(config.ANNOTATED_VIDEOS_PATH, video_name)
                shutil.move(video_path, destination)
                messagebox.showinfo("Success", f"Moved {video_name} to annotated folder.")
            else:
                if video_name.endswith('/'):
                    folder_name = video_name[:-1]
                    destination = os.path.join(config.UNANNOTATED_VIDEOS_PATH, folder_name)
                else:
                    destination = os.path.join(config.UNANNOTATED_VIDEOS_PATH, video_name)
                shutil.move(video_path, destination)
                messagebox.showinfo("Success", f"Moved {video_name} to unannotated folder.")
                
            self.selected_video_info = None
            self.selection_label.config(text="No video selected")
            self.load_video_lists()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to move file: {str(e)}")

    def load_jpeg_sequence(self, folder_path):
        """Load JPEG sequence from folder"""
        try:
            if not os.path.isdir(folder_path):
                return False
                
            # Get all JPEG files and sort them numerically
            jpeg_files = [f for f in os.listdir(folder_path) 
                         if f.lower().endswith(('.jpg', '.jpeg'))]
            
            if not jpeg_files:
                return False
                
            # Sort numerically (assuming format like 0001.jpg, 0002.jpg, etc.)
            jpeg_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or '0'))
            
            # Create full paths
            self.jpeg_file_list = [os.path.join(folder_path, f) for f in jpeg_files]
            self.total_frames = len(self.jpeg_file_list)
            self.is_jpeg_sequence = True
            
            print(f"Loaded JPEG sequence: {len(self.jpeg_file_list)} frames from {folder_path}")
            return True
            
        except Exception as e:
            print(f"Failed to load JPEG sequence: {str(e)}")
            return False
                
    def load_video(self, video_path):
        """Load video file or JPEG sequence"""
        try:
            # Clean up previous video
            if self.video_capture:
                self.video_capture.release()
                
            # Reset variables
            self.is_jpeg_sequence = False
            self.jpeg_file_list = []
            
            # Check if it's a folder (JPEG sequence) or file (video)
            if os.path.isdir(video_path):
                # Try to load as JPEG sequence
                if not self.load_jpeg_sequence(video_path):
                    messagebox.showerror("Error", f"Could not load JPEG sequence: {video_path}")
                    return
            else:
                # Load as video file
                self.video_capture = cv2.VideoCapture(video_path)
                
                if not self.video_capture.isOpened():
                    messagebox.showerror("Error", f"Could not open video: {video_path}")
                    return
                    
                self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                
            self.current_video_path = video_path
            self.current_frame_number = 0
            self.is_playing = False
            
            # Reset timestamps when loading new video
            self.turn_timestamps = []
            self.contraction_timestamps = []
            self.reset_counters()
            
            self.play_pause_btn.config(text='Play')
            self.update_frame_display()
            
            # Update frame slider range
            if hasattr(self, 'frame_slider'):
                self.frame_slider.config(to=max(self.total_frames - 1, 0))
                self.frame_slider_var.set(0)
            
            # Update video title
            if hasattr(self, 'video_title_label'):
                display_name = os.path.basename(video_path)
                if display_name.endswith('/'):
                    display_name = display_name[:-1]  # Remove trailing slash
                display_title = self.extract_video_title(display_name)
                self.video_title_label.config(text=display_title)
            
            print(f"Loaded: {video_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load: {str(e)}")

    def load_next_unannotated_video(self):
        """Load the first video from the unannotated list"""
        if len(self.unannotated_original_names) > 0:
            # Get first video
            next_video_name = self.unannotated_original_names[0]
            
            # Check if it's a JPEG folder or video file
            if next_video_name.endswith('/'):
                folder_name = next_video_name[:-1]
                next_video_path = os.path.join(config.UNANNOTATED_VIDEOS_PATH, folder_name)
            else:
                next_video_path = os.path.join(config.UNANNOTATED_VIDEOS_PATH, next_video_name)
            
            # Load the video
            self.load_video(next_video_path)
            
            # Update selection
            self.selected_video_info = (next_video_path, next_video_name, 'unannotated')
            display_name = self.extract_video_title(next_video_name)
            self.selection_label.config(text=f"Selected: {display_name}")
            
            # Highlight in listbox
            self.unannotated_listbox.selection_clear(0, tk.END)
            self.unannotated_listbox.selection_set(0)
            self.unannotated_listbox.see(0)
        else:
            messagebox.showinfo("Complete", "No more unannotated videos!")

    def on_frame_slider_change(self, value):
        """Handle frame slider movement"""
        has_media = self.video_capture or self.is_jpeg_sequence
        if has_media and not self.is_playing:
            new_frame = int(float(value))
            if new_frame != self.current_frame_number:
                self.current_frame_number = new_frame
                self.update_frame_display()

    def update_frame_display(self):
        """Update the video frame display with 90-degree clockwise rotation and adjustments"""
        has_media = self.video_capture or self.is_jpeg_sequence
        if not has_media:
            self.video_canvas.delete("all")
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()
            if canvas_width > 1 and canvas_height > 1:
                self.video_canvas.create_text(canvas_width//2, canvas_height//2,
                                            text="No video loaded.", 
                                            fill="white", font=('Arial', 16))
            return
            
        # Read frame from video or JPEG sequence
        if self.is_jpeg_sequence:
            if 0 <= self.current_frame_number < len(self.jpeg_file_list):
                frame = cv2.imread(self.jpeg_file_list[self.current_frame_number])
                ret = frame is not None
            else:
                ret = False
        else:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
            ret, frame = self.video_capture.read()
        
        if ret:
            # Rotate frame 90 degrees clockwise
            frame_rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame_rgb = cv2.cvtColor(frame_rotated, cv2.COLOR_BGR2RGB)
            
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                frame_height, frame_width = frame_rgb.shape[:2]
                
                scale_x = canvas_width / frame_width
                scale_y = canvas_height / frame_height
                scale = min(scale_x, scale_y)
                
                new_width = int(frame_width * scale)
                new_height = int(frame_height * scale)
                
                frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
                
                # Apply display adjustments
                pil_image = Image.fromarray(frame_resized)
                
                if self.brightness_adjustment != 0:
                    enhancer = ImageEnhance.Brightness(pil_image)
                    brightness_factor = 1 + (self.brightness_adjustment / 100)
                    pil_image = enhancer.enhance(brightness_factor)
                
                if self.contrast_adjustment != 0:
                    enhancer = ImageEnhance.Contrast(pil_image)
                    contrast_factor = 1 + (self.contrast_adjustment / 100)
                    pil_image = enhancer.enhance(contrast_factor)
                
                self.current_frame_image = ImageTk.PhotoImage(pil_image)
                
                self.video_canvas.delete("all")
                x = (canvas_width - new_width) // 2
                y = (canvas_height - new_height) // 2
                self.video_canvas.create_image(x, y, anchor='nw', image=self.current_frame_image)
        
        self.frame_label.config(text=f'Frame: {self.current_frame_number + 1} / {self.total_frames}')
        
        # Update frame slider without triggering callback
        if hasattr(self, 'frame_slider_var'):
            self.frame_slider_var.set(self.current_frame_number)
        
        # Update progress bar marks
        self.update_progress_bar_marks()
        
    def update_progress_bar_marks(self):
        """Update the colored marks above the frame slider"""
        if not hasattr(self, 'marks_canvas') or self.total_frames == 0:
            return
            
        # Clear existing marks
        self.marks_canvas.delete("timestamp_mark")
        
        canvas_width = self.marks_canvas.winfo_width()
        canvas_height = self.marks_canvas.winfo_height()
        
        if canvas_width <= 1:
            return
            
        # Draw turn marks (turquoise/cyan)
        for frame in self.turn_timestamps:
            x_pos = (frame / max(self.total_frames - 1, 1)) * canvas_width
            self.marks_canvas.create_line(x_pos, 0, x_pos, canvas_height,
                                         fill='#40E0D0', width=3, tags="timestamp_mark")
        
        # Draw contraction marks (soft orange)  
        for frame in self.contraction_timestamps:
            x_pos = (frame / max(self.total_frames - 1, 1)) * canvas_width
            self.marks_canvas.create_line(x_pos, 0, x_pos, canvas_height,
                                         fill='#FFA07A', width=3, tags="timestamp_mark")
        
    def toggle_play_pause(self):
        """Toggle between play and pause"""
        has_media = self.video_capture or self.is_jpeg_sequence
        if not has_media:
            return
            
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            self.play_pause_btn.config(text='Pause')
            self.play_video()
        else:
            self.play_pause_btn.config(text='Play')
            
    def play_video(self):
        """Play video continuously"""
        # Check if we have either video capture or JPEG sequence loaded
        has_media = self.video_capture or self.is_jpeg_sequence
        if self.is_playing and has_media and self.current_frame_number < self.total_frames - 1:
            self.current_frame_number += 1
            self.update_frame_display()
            self.root.after(33, self.play_video)
        else:
            self.is_playing = False
            self.play_pause_btn.config(text='Play')
            
    def frame_forward(self):
        """Go to next frame"""
        has_media = self.video_capture or self.is_jpeg_sequence
        if has_media and self.current_frame_number < self.total_frames - 1:
            self.current_frame_number += 1
            self.update_frame_display()
            
    def frame_backward(self):
        """Go to previous frame"""
        has_media = self.video_capture or self.is_jpeg_sequence
        if has_media and self.current_frame_number > 0:
            self.current_frame_number -= 1
            self.update_frame_display()
        
    def create_video_panel(self, parent):
        """Top right panel for video player - now larger"""
        video_frame = ttk.LabelFrame(parent, text='Video Player', padding=5)
        video_frame.grid(row=0, column=0, sticky='nsew', pady=(0,5))
        
        video_frame.grid_columnconfigure(0, weight=1)
        video_frame.grid_rowconfigure(1, weight=1)  # Canvas gets the weight
        
        # Video title at the top
        self.video_title_label = tk.Label(video_frame, text='No Video Loaded', 
                                         font=('Arial', 14, 'bold'), 
                                         fg='white', bg='black', 
                                         pady=10)
        self.video_title_label.grid(row=0, column=0, sticky='ew', padx=5, pady=(5,10))
        
        # Video canvas - now gets more space
        self.video_canvas = tk.Canvas(video_frame, bg='black', height=500)
        self.video_canvas.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        
        # Controls - centered
        controls_frame = tk.Frame(video_frame)
        controls_frame.grid(row=2, column=0, padx=5, pady=5)
        
        frame_back_btn = tk.Button(controls_frame, text='◀', width=3,
                                  command=self.frame_backward, font=('Arial', 12))
        frame_back_btn.pack(side='left', padx=5)
        
        self.play_pause_btn = tk.Button(controls_frame, text='Play',
                                       command=self.toggle_play_pause, 
                                       font=('Arial', 12), width=8)
        self.play_pause_btn.pack(side='left', padx=5)
        
        frame_forward_btn = tk.Button(controls_frame, text='▶', width=3,
                                     command=self.frame_forward, font=('Arial', 12))
        frame_forward_btn.pack(side='left', padx=5)

        # Display options button
        tk.Button(controls_frame, text='Video Display Options',
                 command=self.show_display_options, font=('Arial', 10),
                 width=15).pack(side='left', padx=10)
        
        # Frame slider and marks section
        slider_frame = tk.Frame(video_frame)
        slider_frame.grid(row=3, column=0, sticky='ew', padx=5, pady=(0,5))
        slider_frame.grid_columnconfigure(0, weight=1)
        
        # Canvas for timestamp marks above slider - match slider padding exactly
        self.marks_canvas = tk.Canvas(slider_frame, height=15, highlightthickness=0)
        self.marks_canvas.grid(row=0, column=0, sticky='ew', padx=(10,90))
        self.marks_canvas.bind('<Configure>', lambda e: self.update_progress_bar_marks())
        
        # Frame slider
        self.frame_slider_var = tk.DoubleVar()
        self.frame_slider = tk.Scale(slider_frame, from_=0, to=100, orient='horizontal',
                                    variable=self.frame_slider_var, command=self.on_frame_slider_change,
                                    showvalue=False, length=300)
        self.frame_slider.grid(row=1, column=0, sticky='ew', padx=(0,80))
        
        self.frame_label = tk.Label(slider_frame, text='Frame: 0 / 0', font=('Arial', 10))
        self.frame_label.grid(row=1, column=1)
        
        self.update_frame_display()


    def create_scoring_panel(self, parent):
        """Bottom right panel for scoring interface - more compact"""
        scoring_frame = tk.Frame(parent, relief='raised', borderwidth=1)
        scoring_frame.grid(row=1, column=0, sticky='nsew', padx=2, pady=2)
        
        scoring_frame.grid_columnconfigure(0, weight=1)
        
        # Counters section - more compact
        counters_frame = tk.Frame(scoring_frame)
        counters_frame.grid(row=0, column=0, pady=8)
        
        # TURNS counter
        turns_frame = tk.Frame(counters_frame)
        turns_frame.grid(row=0, column=0, padx=20)
        
        tk.Label(turns_frame, text='TURNS', font=('Arial', 12, 'bold')).pack()
        
        self.turns_count = tk.StringVar(value='0')
        self.turns_btn = tk.Button(turns_frame, textvariable=self.turns_count, 
                                  font=('Arial', 20, 'bold'),
                                  width=4, height=1,
                                  command=lambda: self.increment_counter('turns'),
                                  bg='#40E0D0')  # Turquoise
        self.turns_btn.pack(pady=3)
        
        # CONTRACTIONS counter
        contractions_frame = tk.Frame(counters_frame)
        contractions_frame.grid(row=0, column=1, padx=20)
        
        tk.Label(contractions_frame, text='CONTRACTIONS', font=('Arial', 12, 'bold')).pack()
        
        self.contractions_count = tk.StringVar(value='0')
        self.contractions_btn = tk.Button(contractions_frame, textvariable=self.contractions_count,
                                         font=('Arial', 20, 'bold'),
                                         width=4, height=1,
                                         command=lambda: self.increment_counter('contractions'),
                                         bg='#FFA07A')  # Soft orange
        self.contractions_btn.pack(pady=3)
        
        # Control buttons section - more compact
        controls_section = tk.Frame(scoring_frame)
        controls_section.grid(row=1, column=0, pady=(5, 8))
        
        # Profile selection and save - compact
        profile_controls = tk.Frame(controls_section)
        profile_controls.grid(row=0, column=0, pady=(0, 5))
        
        tk.Label(profile_controls, text='PROFILE:', font=('Arial', 10, 'bold')).pack(side='left', padx=(0, 5))
        
        self.profile_var = tk.StringVar(value=getattr(config, 'PROFILES', ["Default"])[0])
        profile_dropdown = ttk.Combobox(profile_controls, textvariable=self.profile_var,
                                       values=getattr(config, 'PROFILES', ["Default"]),
                                       state='readonly', width=12, font=('Arial', 9))
        profile_dropdown.pack(side='left', padx=(0, 8))
        
        # CSV save dropdown with refresh
        csv_files = ["New file"] + self.get_csv_files()
        self.csv_file_var = tk.StringVar(value="New file")
        self.csv_dropdown = ttk.Combobox(profile_controls, textvariable=self.csv_file_var,
                                        values=csv_files, state='readonly', width=12, font=('Arial', 9))
        self.csv_dropdown.pack(side='left', padx=(0, 8))
        
        # Refresh CSV list button
        tk.Button(profile_controls, text='↻', command=self.refresh_csv_dropdown,
                 font=('Arial', 10), width=2).pack(side='left', padx=(0, 8))
        
        # Expand button for CSV selection - NEW
        tk.Button(profile_controls, text='Expand', command=self.show_expanded_save_options,
                 font=('Arial', 9), width=8).pack(side='left', padx=(0, 8))
        
        # Create new copy checkbox
        self.create_copy_var = tk.BooleanVar()
        tk.Checkbutton(profile_controls, text='Create copy', variable=self.create_copy_var,
                      font=('Arial', 9)).pack(side='left')
        
        # Notes section - compact
        notes_section = tk.Frame(controls_section)
        notes_section.grid(row=1, column=0, pady=(3, 5), sticky='ew')
        controls_section.grid_columnconfigure(0, weight=1)
        
        tk.Label(notes_section, text='NOTES:', font=('Arial', 10, 'bold')).pack(anchor='w')
        
        self.notes_text = tk.Text(notes_section, height=2, width=50, font=('Arial', 9),
                                 wrap='word', relief='sunken', borderwidth=1)
        self.notes_text.pack(fill='x', pady=2)
        
        # Bind focus events to handle keyboard shortcuts
        self.notes_text.bind('<FocusIn>', self.on_notes_focus_in)
        self.notes_text.bind('<FocusOut>', self.on_notes_focus_out)
        self.notes_text.bind('<Escape>', self.on_notes_escape)
        
        # Other control buttons - compact
        button_frame = tk.Frame(controls_section)
        button_frame.grid(row=2, column=0, pady=(3, 0))
        
        tk.Button(button_frame, text='Reset', command=self.confirm_reset,
                 font=('Arial', 9), width=8).pack(side='left', padx=2)
        
        tk.Button(button_frame, text='Save', command=self.save_data,
                 font=('Arial', 10, 'bold'), bg='lightgreen', width=10).pack(side='left', padx=10)
        
        tk.Button(button_frame, text='Reload', command=self.reload_video,
                 font=('Arial', 9), width=8).pack(side='left', padx=2)
        
        # Instructions button at bottom
        tk.Button(controls_section, text='Instructions', command=self.show_help,
                 font=('Arial', 10), width=12, bg='lightgray').grid(row=3, column=0, pady=(8, 0))
        
    def increment_counter(self, counter_type):
        """Increment behavior counter and mark current frame"""
        current_frame = self.current_frame_number
        
        if counter_type == 'turns':
            # Only add if not already present at this frame
            if current_frame not in self.turn_timestamps:
                current = int(self.turns_count.get())
                current += 1
                self.turns_count.set(str(current))
                self.turn_timestamps.append(current_frame)
                self.update_progress_bar_marks()
        else:
            # Only add if not already present at this frame
            if current_frame not in self.contraction_timestamps:
                current = int(self.contractions_count.get())
                current += 1
                self.contractions_count.set(str(current))
                self.contraction_timestamps.append(current_frame)
                self.update_progress_bar_marks()
            
    def decrement_counter(self, counter_type):
        """Decrement behavior counter and remove frame mark if needed"""
        current_frame = self.current_frame_number
        
        if counter_type == 'turns':
            current = int(self.turns_count.get())
            if current > 0:
                current -= 1
                self.turns_count.set(str(current))
                
                # Remove timestamp if present
                if current_frame in self.turn_timestamps:
                    self.turn_timestamps.remove(current_frame)
                    self.update_progress_bar_marks()
        else:
            current = int(self.contractions_count.get())
            if current > 0:
                current -= 1
                self.contractions_count.set(str(current))
                
                # Remove timestamp if present
                if current_frame in self.contraction_timestamps:
                    self.contraction_timestamps.remove(current_frame)
                    self.update_progress_bar_marks()
                    
    def reset_counters(self):
        """Reset all counters, timestamps, and notes to 0"""
        self.turns_count.set('0')
        self.contractions_count.set('0')
        self.turn_timestamps = []
        self.contraction_timestamps = []
        self.notes_text.delete("1.0", tk.END)  # Clear the notes box
        self.update_progress_bar_marks()

    def confirm_reset(self):
        """Show confirmation dialog before resetting"""
        result = messagebox.askyesno("Confirm Reset", 
                                   "Are you sure you want to reset all counters and timestamps?",
                                   icon='question')
        if result:
            self.reset_counters()
        
    def refresh_csv_dropdown(self):
        """Refresh the CSV file dropdown with current files"""
        csv_files = ["New file"] + self.get_csv_files()
        self.csv_dropdown['values'] = csv_files
        
    def save_data(self):
        """Show save confirmation dialog with CSV functionality"""
        has_media = self.video_capture or self.is_jpeg_sequence
        if not has_media:
            messagebox.showwarning("No Video", "Please load a video first.")
            return
            
        # Check if video already annotated in selected CSV
        filename = os.path.basename(self.current_video_path)
        # For JPEG folders, use folder name
        if self.is_jpeg_sequence:
            filename = os.path.basename(self.current_video_path) + "/"
            
        already_annotated = self.check_video_already_annotated(self.csv_file_var.get(), filename)
        
        # Create confirmation dialog
        self.save_dialog = tk.Toplevel(self.root)
        self.save_dialog.title("Confirm Save")
        self.save_dialog.geometry("450x350")
        self.save_dialog.transient(self.root)
        self.save_dialog.grab_set()
        
        # Center the dialog
        self.save_dialog.geometry("+%d+%d" % (self.root.winfo_rootx() + 50, self.root.winfo_rooty() + 50))
        
        # Warning for already annotated videos
        if already_annotated:
            warning_frame = tk.Frame(self.save_dialog, bg='red', relief='raised', borderwidth=2)
            warning_frame.pack(fill='x', padx=10, pady=10)
            tk.Label(warning_frame, text="WARNING: Video already annotated in selected CSV file! ⚠️", 
                    font=('Arial', 11, 'bold'), bg='red', fg='white').pack(pady=5)
        
        tk.Label(self.save_dialog, text="Are you sure you want to save?", 
                font=('Arial', 12, 'bold')).pack(pady=10)
        
        # Show video name and profile
        display_title = self.extract_video_title(filename)
        
        info_frame = tk.Frame(self.save_dialog)
        info_frame.pack(pady=10)
        
        tk.Label(info_frame, text=f"Video: {display_title}", font=('Arial', 11)).pack()
        tk.Label(info_frame, text=f"Profile: {self.profile_var.get()}", font=('Arial', 11)).pack()
        tk.Label(info_frame, text=f"Turns: {self.turns_count.get()}", font=('Arial', 11)).pack()
        tk.Label(info_frame, text=f"Contractions: {self.contractions_count.get()}", font=('Arial', 11)).pack()
        tk.Label(info_frame, text=f"CSV File: {self.csv_file_var.get()}", font=('Arial', 11)).pack()
        
        # Show timestamps
        turn_frames = ",".join(map(str, sorted(self.turn_timestamps))) if self.turn_timestamps else "None"
        contraction_frames = ",".join(map(str, sorted(self.contraction_timestamps))) if self.contraction_timestamps else "None"
        
        tk.Label(info_frame, text=f"Turn Timestamps: {turn_frames}", font=('Arial', 9)).pack()
        tk.Label(info_frame, text=f"Contraction Timestamps: {contraction_frames}", font=('Arial', 9)).pack()
        
        # Buttons
        button_frame = tk.Frame(self.save_dialog)
        button_frame.pack(pady=15)
        
        tk.Button(button_frame, text='Yes (Y)', command=self.confirm_save,
                 font=('Arial', 11, 'bold'), bg='lightgreen', width=10).pack(side='left', padx=10)
        tk.Button(button_frame, text='No (N)', command=self.cancel_save,
                 font=('Arial', 11, 'bold'), bg='lightcoral', width=10).pack(side='left', padx=10)
        
        # Bind keyboard shortcuts directly to the dialog
        self.save_dialog.bind('<Key-y>', lambda e: self.confirm_save())
        self.save_dialog.bind('<Key-Y>', lambda e: self.confirm_save())
        self.save_dialog.bind('<Key-n>', lambda e: self.cancel_save())
        self.save_dialog.bind('<Key-N>', lambda e: self.cancel_save())
        self.save_dialog.bind('<Return>', lambda e: self.confirm_save())  # Enter to confirm
        self.save_dialog.bind('<Escape>', lambda e: self.cancel_save())   # Escape to cancel
        
        # Focus on dialog for keyboard input
        self.save_dialog.focus_set()
        
    def confirm_save(self):
        """Confirm and execute save to CSV"""
        try:
            # Prepare data
            filename = os.path.basename(self.current_video_path)
            # For JPEG folders, use folder name with trailing slash
            if self.is_jpeg_sequence:
                filename = os.path.basename(self.current_video_path) + "/"
                
            turn_frames = ",".join(map(str, sorted(self.turn_timestamps))) if self.turn_timestamps else ""
            contraction_frames = ",".join(map(str, sorted(self.contraction_timestamps))) if self.contraction_timestamps else ""
            notes_content = self.notes_text.get("1.0", tk.END).strip()
            
            # Determine CSV file path
            if self.csv_file_var.get() == "New file":
                csv_filename = self.generate_csv_filename()
                csv_path = os.path.join(config.CSV_SAVE_PATH, csv_filename)
                file_exists = False
            else:
                csv_filename = self.csv_file_var.get()
                csv_path = os.path.join(config.CSV_SAVE_PATH, csv_filename)
                file_exists = os.path.exists(csv_path)
                
                # Handle "Create new copy" option
                if self.create_copy_var.get() and file_exists:
                    new_filename = self.generate_csv_filename()
                    new_csv_path = os.path.join(config.CSV_SAVE_PATH, new_filename)
                    shutil.copy2(csv_path, new_csv_path)
                    csv_path = new_csv_path
                    csv_filename = new_filename
                    file_exists = False
            
            # Prepare row data
            row_data = [
                filename,  # Video filename
                "",        # TimestampID (left blank)
                self.turns_count.get(),  # Number_turns
                turn_frames,  # Turn_timestamps
                self.contractions_count.get(),  # Number_contractions
                contraction_frames,  # Contraction_timestamps
                notes_content  # Notes
            ]
            
            # Write to CSV
            with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header if new file
                if not file_exists:
                    header = ["Video_filename", "TimestampID", "Number_turns", "Turn_timestamps", 
                             "Number_contractions", "Contraction_timestamps", "Notes"]
                    writer.writerow(header)
                
                # Write data row
                writer.writerow(row_data)
            
            self.save_dialog.destroy()
            messagebox.showinfo("Saved", f"Data saved to {csv_filename}!")
            
            # Refresh CSV dropdown
            self.refresh_csv_dropdown()
            
            # Move file to Annotated folder after successful save
            self.move_current_video_to_annotated()
            
            # Auto-load next video
            self.load_next_unannotated_video()
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save data: {str(e)}")
    
    def move_current_video_to_annotated(self):
        """Move the currently loaded video to the Annotated folder"""
        if not self.current_video_path:
            return
        
        try:
            # Determine if current video is in Unannotated folder
            if config.UNANNOTATED_VIDEOS_PATH in self.current_video_path:
                # Get the filename or folder name
                if self.is_jpeg_sequence:
                    folder_name = os.path.basename(self.current_video_path)
                    destination = os.path.join(config.ANNOTATED_VIDEOS_PATH, folder_name)
                    display_name = folder_name + "/"
                else:
                    video_name = os.path.basename(self.current_video_path)
                    destination = os.path.join(config.ANNOTATED_VIDEOS_PATH, video_name)
                    display_name = video_name
                
                # Move the file/folder
                shutil.move(self.current_video_path, destination)
                
                # Update current path
                self.current_video_path = destination
                
                # Clear selection
                if self.selected_video_info:
                    self.selected_video_info = None
                    self.selection_label.config(text="No video selected")
                
                # Refresh lists
                self.load_video_lists()
                
                print(f"Moved {display_name} to Annotated folder after save.")
                
        except Exception as e:
            print(f"Note: Could not move file to Annotated folder: {str(e)}")
            # Don't show error dialog - file was already saved successfully
            
    def cancel_save(self):
        """Cancel save operation"""
        self.save_dialog.destroy()
        
    def reload_video(self):
        """Reload current video"""
        if self.current_video_path:
            self.load_video(self.current_video_path)
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = VideoAnnotationInterface()
    app.run()