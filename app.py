# Import necessary libraries
import gradio as gr
import cv2
import os
import numpy as np
import shutil
import json
import av 
import torch
import requests
import threading
from packaging import version
from collections import namedtuple
from fractions import Fraction
from tqdm import tqdm
from datetime import datetime
from sammie.smooth import run_smoothing_model, prepare_smoothing_model
from matanyone.inference.inference_core import InferenceCore
from matanyone.utils.get_default_model import get_matanyone_model
from sammie.duplicate_frame_handler import replace_similar_matte_frames

# .........................................................................................
# Global variables
# .........................................................................................
__version__ = "1.6.1"
temp_dir = "temp"
frames_dir = os.path.join(temp_dir, "frames")
mask_dir = os.path.join(temp_dir, "masks")
matting_dir = os.path.join(temp_dir, "matting")
settings = None
session = None
edge_smoothing = False
device = None
propagated = False # whether we have propagated the masks
propagating = False # whether we are currently propagating
inference_state = None
predictor = None
mat_processor = None
points_list = []
DataPoint = namedtuple("DataPoint", ["Frame", "ObjectID", "Positive", "X", "Y"])
PALETTE = [
    (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
    (128, 128, 128), (64, 0, 0), (191, 0, 0), (64, 128, 0), (191, 128, 0), (64, 0, 128),
    (191, 0, 128), (64, 128, 128), (191, 128, 128), (0, 64, 0), (128, 64, 0), (0, 191, 0),
    (128, 191, 0), (0, 64, 128), (128, 64, 128)
]

# .........................................................................................
# Functions
# .........................................................................................

# Set up CUDA if available
def setup_cuda():
    print(f"Sammie-Roto version {__version__}")
    # if using Apple MPS, fall back to CPU for unsupported ops
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    print("PyTorch version:", torch.__version__)

    # select the device for computation
    if settings.get("force_cpu"):
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        print("CUDA Compute Capability: ", torch.cuda.get_device_capability())
        # turn on tfloat32 for Ampere GPUs / not sure if this has an effect, may remove in the future
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    return device

# Load the model and predictor / this can only be called once because the imports will interfere with each other
def load_model():
    clear_cache()
    model_selection = settings.get("segmentation_model")
    if (model_selection == "auto" and device.type == "cpu") or model_selection == "efficient":
        from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor
        print("Using EfficientTAM model")
        checkpoint = "./checkpoints/efficienttam_s_512x512.pt"
        model_cfg = "../configs/efficienttam_s_512x512.yaml"
        return build_efficienttam_video_predictor(model_cfg, checkpoint, device=device)
    if model_selection == "sam2base" or (model_selection == "auto" and torch.cuda.is_available() and torch.cuda.get_device_properties(0).major < 8):
        from sam2.build_sam import build_sam2_video_predictor
        print("Using SAM2 Base model")
        checkpoint = "./checkpoints/sam2.1_hiera_base_plus.pt"
        model_cfg = "../configs/sam2.1_hiera_b+.yaml"
        return build_sam2_video_predictor(model_cfg, checkpoint, device=device) 
    else:
        from sam2.build_sam import build_sam2_video_predictor
        print("Using SAM2 Large model")
        checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "../configs/sam2.1_hiera_l.yaml"
        return build_sam2_video_predictor(model_cfg, checkpoint, device=device) 

def load_matting_model():
    checkpoint = "./checkpoints/matanyone.pth"
    matanyone = get_matanyone_model(checkpoint, device=device)
    print("Loaded MatAnyone model")
    # init inference processor
    return InferenceCore(matanyone, cfg=matanyone.cfg, device=device)

# Check for updates
def start_update_check(repo="Zarxrax/Sammie-Roto", timeout=5):
    def background_check():
        try:
            url = f"https://api.github.com/repos/{repo}/releases/latest"
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                latest = response.json().get("tag_name", "").lstrip("v")
                if version.parse(latest) > version.parse(__version__):
                    print(f"\n🔔 A new version of Sammie-roto is available! ({__version__} → {latest})")
                    print(f"👉 Download: https://github.com/{repo}/releases\n")
        except Exception:
            pass  # Fail silently, no impact on startup
    threading.Thread(target=background_check, daemon=True).start()

# Save settings if the user changes the settings
def change_settings(model_dropdown, matting_quality_dropdown, cpu_checkbox):
    if model_dropdown == "Auto":
        settings["segmentation_model"] = "auto"
    elif model_dropdown == "SAM2.1Large (High Quality)":
        settings["segmentation_model"] = "sam2large"
    elif model_dropdown == "SAM2.1Base+":
        settings["segmentation_model"] = "sam2base"
    elif model_dropdown == "EfficientTAM (Fast)":
        settings["segmentation_model"] = "efficient"
    if matting_quality_dropdown == "480p":
        settings["matting_quality"] = 480
    elif matting_quality_dropdown == "720p":
        settings["matting_quality"] = 720
    elif matting_quality_dropdown == "1080p":
        settings["matting_quality"] = 1080
    elif matting_quality_dropdown == "Full":
        settings["matting_quality"] = -1
    if cpu_checkbox:
        settings["force_cpu"] = True
    else:
        settings["force_cpu"] = False
    save_settings()
    gr.Info("You must restart the application for changes to take effect.")

def change_export_settings(export_type_dropdown, export_content_dropdown, export_object_dropdown):
    settings["export_type"] = export_type_dropdown
    settings["export_content"] = export_content_dropdown
    settings["export_object"] = export_object_dropdown
    save_settings()

# Save settings if user changes the postprocessing settings
def change_postprocessing(post_holes_slider, post_dots_slider, post_grow_slider, post_border_slider, show_outlines_checkbox, post_gamma_slider, post_grow_matte_slider):
    settings["holes"] = post_holes_slider
    settings["dots"] = post_dots_slider
    settings["grow"] = post_grow_slider
    settings["border"] = post_border_slider
    settings["show_outlines"] = show_outlines_checkbox
    settings["gamma"] = post_gamma_slider
    settings["grow_matte"] = post_grow_matte_slider
    save_settings()

def reset_postprocessing():
    settings["holes"] = 0
    settings["dots"] = 0
    settings["grow"] = 0
    settings["border"] = 0
    settings["gamma"] = 1
    settings["grow_matte"] = 0
    settings["show_outlines"] = True
    save_settings()
    return [gr.Slider(minimum=0, maximum=50, value=0, step=1, label="Remove Holes"), gr.Slider(minimum=0, maximum=50, value=0, step=1, label="Remove Dots"), gr.Slider(minimum=-10, maximum=10, value=0, step=1, label="Shrink/Grow"), gr.Slider(minimum=0, maximum=10, value=0, step=1, label="Border Fix"), gr.Checkbox(label="Show Outlines", value=True, interactive=True), gr.Slider(minimum=0.01, maximum=10, value=1, step=0.01, label="Gamma"), gr.Slider(minimum=-10, maximum=10, value=0, step=1, label="Shrink/Grow")]

# Load settings from json file
def load_settings():
    default_settings = {
    "segmentation_model": "auto",
    "matting_quality": 720,
    "force_cpu": False,
    "export_fps": 24,
    "export_type": "Matte",
    "export_content": "Segmentation Mask",
    "export_object": "All",
    "holes": 0,
    "dots": 0,
    "grow": 0,
    "border": 0,
    "gamma": 0,
    "grow_matte": 0,
    "show_outlines": True,
    "dedupe_min_threshold" : 0.8
    }
    try:
        with open("settings.json", 'r') as file:
            user_settings = json.load(file)
            # Merge defaults with user settings
            merged_settings = {**default_settings, **user_settings}
            return merged_settings
    except FileNotFoundError:
        print("Settings file not found. Using default settings.")
        return default_settings
    except json.JSONDecodeError:
        print("Error decoding JSON. Using default settings.")
        return default_settings
    
# Save settings to json file
def save_settings():
    try:
        with open("settings.json", 'w') as file:
            json.dump(settings, file, indent=4)
    except Exception as e:
        print(f"Error saving settings: {e}")

# Load session from json file (create if missing)
def load_session():
    default_session = {
        "input_file_name": os.path.join(temp_dir, "output"),
        "name_roto": True,
        "name_object": False,
        "name_type": False,
        "name_date_time": False
    }
    try:
        with open("session.json", 'r') as file:
            user_session = json.load(file)
            merged_session = {**default_session, **user_session}
            return merged_session
    except FileNotFoundError:
        # Create session.json with defaults
        try:
            with open("session.json", 'w') as file:
                json.dump(default_session, file, indent=4)
        except Exception as e:
            print(f"Error creating session.json: {e}")
        return default_session
    except json.JSONDecodeError:
        print("Error decoding session.json. Using default session values.")
        return default_session

# Save session to json file
def save_session():
    try:
        with open("session.json", 'w') as file:
            json.dump(session, file, indent=4)
    except Exception as e:
        print(f"Error saving session: {e}")

# Setup the values for the model dropdown box based on the settings file
def set_model_dropdown():
    if settings["segmentation_model"] == "auto":
        return "Auto"
    elif settings["segmentation_model"] == "sam2large":
        return "SAM2.1Large (High Quality)"
    elif settings["segmentation_model"] == "sam2base":
        return "SAM2.1Base+"
    elif settings["segmentation_model"] == "efficient":
        return "EfficientTAM (Fast)"

# Function to build video filename based on session settings and export options
def build_video_filename():
    global session
    
    # Start with input filename
    filename_parts = [session.get("input_file_name", "output")]
    
    # Add roto if enabled
    if session.get("name_roto", True):
        filename_parts.append("SammieRoto")
    
    # Add type if enabled
    if session.get("name_type", False):
        export_type = settings.get("export_type", "Matte")
        filename_parts.append(export_type)
    
    # Add object if enabled
    if session.get("name_object", False):
        export_object = settings.get("export_object", "All")
        export_object = str(export_object)  # Ensure it's a string
        filename_parts.append(export_object)
    
    # Add date & time if enabled
    if session.get("name_date_time", False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_parts.append(timestamp)
    
    # Join all parts with underscores
    return "_".join(filename_parts)

# Functions to handle name option checkbox changes
def update_name_roto(checked):
    global session
    session["name_roto"] = checked
    save_session()
    return build_video_filename()

def update_name_type(checked):
    global session
    session["name_type"] = checked
    save_session()
    return build_video_filename()

def update_name_object(checked):
    global session
    session["name_object"] = checked
    save_session()
    return build_video_filename()

def update_name_date_time(checked):
    global session
    session["name_date_time"] = checked
    save_session()
    return build_video_filename()

# Setup the values for the matting quality dropdown box based on the settings file
def set_matting_quality_dropdown():
    if settings["matting_quality"] == 480:
        return "480p"
    elif settings["matting_quality"] == 720:
        return "720p"
    elif settings["matting_quality"] == 1080:
        return "1080p"
    elif settings["matting_quality"] == -1:
        return "Full"

def set_export_type_dropdown():
    if settings["export_type"] == "Matte": 
        return "Matte"
    elif settings["export_type"] == "Alpha": 
        return "Alpha"
    elif settings["export_type"] == "Greenscreen": 
        return "Greenscreen"

def set_export_content_dropdown():
    if settings["export_content"] == "Segmentation Mask": 
        return "Segmentation Mask"
    elif settings["export_content"] == "Segmentation with Edge Smoothing": 
        return "Segmentation with Edge Smoothing"
    elif settings["export_content"] == "Matting": 
        return "Matting"

def set_postprocessing_holes_slider():
    return settings["holes"]

def set_postprocessing_dots_slider():
    return settings["dots"]

def set_postprocessing_grow_slider():
    return settings["grow"]

def set_postprocessing_border_slider():
    return settings["border"]

def set_postprocessing_gamma_slider():
    return settings["gamma"]

def set_postprocessing_grow_matte_slider():
    return settings["grow_matte"]

def set_show_outlines():
    return settings["show_outlines"]

# Helper function to clear cache
def clear_cache():
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

# Function to process the video and save frames as PNGs
def process_video(video_file, progress=gr.Progress()):
    global inference_state, session
    inference_state = None # empty out the inference state if its populated

    # Extract filename without extension and store in session
    if video_file and hasattr(video_file, 'name'):
        filename = os.path.basename(video_file.name)
        filename_without_ext = os.path.splitext(filename)[0]
        session["input_file_name"] = filename_without_ext
        save_session()

    # Create temp directories to save the frames, delete if already exists
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(frames_dir)
    os.makedirs(mask_dir)
    os.makedirs(matting_dir)
    
    # Open the video file and initialize frame counter
    progress(0, desc="Extracting Frames...")
    cap = cv2.VideoCapture(video_file.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    settings['export_fps'] = round(float(cap.get(cv2.CAP_PROP_FPS)), 3)
    save_settings()
    if total_frames > 1500:
        gr.Warning('It looks like you are loading a long video. This may take a while and use a lot of disk space. If this is not what you intended, please restart the application and load a shorter video.')
    frame_count = 0
    
    # Read and save frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(frames_dir, f"{frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
        progress(frame_count / total_frames)
    cap.release()

    # initialize the predictor
    inference_state = predictor.init_state(video_path=frames_dir, async_loading_frames=True, offload_video_to_cpu=True)
    progress(1)
    return frame_count

# Function to call process_video and get the frame count, then set up the slider to use that number of frames, also update the fps in the export tab
def process_and_enable_slider(video_file):
    settings['export_object'] = "All"  # reset the export object to All when a new input file is uploaded
    save_settings()
    frame_count = process_video(video_file)
    return [gr.Slider(minimum=0,maximum=frame_count-1, value=0, step=1, label="Frame Number"), gr.Slider(minimum=0,maximum=frame_count-1, value=0, step=1, label="Frame Number"), gr.Dropdown(choices=[23.976, 24, 29.97, 30], value=str(settings['export_fps']), label="FPS", allow_custom_value=True, interactive=True)]

# Function to modify the value of the frame slider when clicking on the dataframe, also update the displayed object color
def change_slider(event_data: gr.SelectData):
    color = '#%02x%02x%02x' % PALETTE[event_data.row_value[1] % len(PALETTE)] # convert color palette to hex
    return event_data.row_value[0], gr.Number(label="Object ID", value=event_data.row_value[1], minimum=0, maximum=20, step=1, interactive=True, min_width=100), gr.ColorPicker(label="Object Color", value=color, interactive=False, min_width=100)

# Function to count the number of frames, used when launching the application to resume previous work
def count_frames():
    if not os.path.exists(frames_dir):
        return 0
    frame_count = len([f for f in os.listdir(frames_dir) if f.endswith(".png")])
    return frame_count

# returns the list of object IDs
def get_objects():
    object_ids = list({point.ObjectID for point in points_list})
    return object_ids

def count_masks():  #count the folders in the masks directory
    if not os.path.exists(mask_dir):
        return 0
    mask_count = len([f for f in os.listdir(mask_dir) if os.path.isdir(os.path.join(mask_dir, f))])
    return mask_count

def save_json(points_list):
    json_filename = os.path.join(temp_dir, "points.json")
    with open(json_filename, 'w') as f:
        json.dump(points_list, f)

# when the user clicks load points button
def load_points(json_filename):
    global points_list
    if os.path.exists(json_filename):
        with open(json_filename, 'r') as f:
            json_data = json.load(f)
            points_list = [DataPoint(*point) for point in json_data]
            save_json(points_list)
    return points_list

# when the user clicks save points button. The file is already defined on the button itself, this function just checks that it exists.
def save_points():
    if not os.path.exists(os.path.join(temp_dir, "points.json")):
        gr.Warning("Please add some points first.")

# Function to update the image based on the slider value
def update_image(slider_value):
    frame_filename = os.path.join(frames_dir, f"{slider_value:04d}.png")
    if os.path.exists(frame_filename):
        image = cv2.imread(frame_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = draw_masks(image, slider_value)
        image = draw_contours(image, slider_value)
        image = draw_points(image, slider_value)
        return image
    else:
        return None

# Function to record points when clicking on the image
def add_point(frame_number, object_id, point_type, event_data: gr.SelectData):
    if not propagating: # dont allow adding points while propagating
        x, y = event_data.index[0], event_data.index[1]
        points_list.append(DataPoint(frame_number, object_id, 1 if point_type == "+" else 0, x, y))
        if propagated:
            clear_tracking()
        segment_image(frame_number, object_id)
        save_json(points_list)
    return points_list

# Function to run segmentation and save the mask
def segment_image(frame_number, object_id):
    frame_filename = os.path.join(frames_dir, f"{frame_number:04d}.png")
    if os.path.exists(frame_filename):

        # format the input points and labels
        filtered_points = [(point.X, point.Y, point.Positive) for point in points_list if point.Frame == frame_number and point.ObjectID == object_id]
        if filtered_points:
            input_points = np.array([(x, y) for x, y, _ in filtered_points], dtype=np.float32)
            input_labels = np.array([positive for _, _, positive in filtered_points], dtype=np.int32)
        else:
            return
        # run the prediction
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box( # returns a list of masks which includes all objects
            inference_state=inference_state,
            frame_idx=frame_number,
            obj_id=object_id,
            points=input_points,
            labels=input_labels
        )
        # Save the segmentation masks
        for i, out_obj_id in enumerate(out_obj_ids):
            mask_filename = os.path.join(mask_dir, f"{frame_number:04d}", f"{out_obj_id}.png")
            mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
            mask = (mask * 255).astype(np.uint8)
            os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
            cv2.imwrite(mask_filename, mask)

# Replay all points, used when masks need to be rebuilt
def replay_points_sequentially():
    frame_count = count_frames()

    for frame_number in range(frame_count):
        # Filter points for the current frame
        frame_points = [point for point in points_list if point.Frame == frame_number]
        if not frame_points:
            continue

        # Group points by ObjectID
        frame_object_ids = {point.ObjectID for point in frame_points}
        for object_id in frame_object_ids:
            # Filter points for the current object ID
            filtered_points = [(point.X, point.Y, point.Positive) 
                               for point in frame_points if point.ObjectID == object_id]

            # Process points incrementally
            for i in range(1, len(filtered_points) + 1):
                # Use only the first `i` points
                subset_points = filtered_points[:i]
                input_points = np.array([(x, y) for x, y, _ in subset_points], dtype=np.float32)
                input_labels = np.array([positive for _, _, positive in subset_points], dtype=np.int32)

                try:
                    # Process the points with the segmentation model
                    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=frame_number,
                        obj_id=object_id,
                        points=input_points,
                        labels=input_labels
                    )

                except Exception as e:
                    print(f"Error during prediction for frame {frame_number}, object {object_id}, points {i}: {e}")
                    continue

                # Save masks
                for j, out_obj_id in enumerate(out_obj_ids):
                    mask_filename = os.path.join(mask_dir, f"{frame_number:04d}", f"{out_obj_id}.png")
                    mask = (out_mask_logits[j] > 0.0).cpu().numpy().squeeze()
                    mask = (mask * 255).astype(np.uint8)
                    try:
                        os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
                        cv2.imwrite(mask_filename, mask)
                    except Exception as e:
                        print(f"Error saving mask for frame {frame_number}, object {out_obj_id}: {e}")


# Draw masks on the current frame
def draw_masks(image, frame_number):
    overlay = image.copy()
    object_ids = get_objects()
    if not object_ids:
        return overlay
    combined_mask = np.zeros_like(image, dtype=np.uint8)  # Initialize combined mask as blank
    mask_found = False  # Track if at least one mask is successfully loaded
    for i, object_id in enumerate(object_ids):
        # Construct the mask file path
        mask_filename = os.path.join(mask_dir, f"{frame_number:04d}", f"{object_id}.png")
        if os.path.exists(mask_filename):
            mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
            if mask is not None and np.any(mask):  # Ensure mask is not blank
                mask_found = True  # A valid mask was found

                # run postprocessing on the mask
                mask = fill_small_holes(mask)
                mask = remove_small_dots(mask)
                mask = grow_shrink(mask)
                mask = border_fix(mask)

                # Get the color from the palette (cycle through if needed)
                color = PALETTE[object_id % len(PALETTE)]
                # Add the current mask to the combined mask
                for c in range(3):  # Apply color to each channel
                    combined_mask[:, :, c] += (mask / 255.0 * color[c]).astype(np.uint8)
    # If no masks were found, return the original image
    if not mask_found:
        return overlay
    # Blend the colored mask with the overlay
    mask_binary = np.any(combined_mask > 0, axis=2)  # Create a binary mask from the combined mask
    if not np.any(mask_binary):  # Ensure the combined mask is not entirely blank
        return overlay
    overlay[mask_binary] = cv2.addWeighted(image[mask_binary], 0.5, combined_mask[mask_binary], 0.5, 0)
    return overlay

# Draw contours on the current frame
def draw_contours(image, frame_number):
    object_ids = get_objects()
    
    # Create a copy of the image to draw the contours on
    image_copy = image.copy()

    if settings["show_outlines"]:
        for i, object_id in enumerate(object_ids):
            # Construct the mask file path
            mask_filename = os.path.join(mask_dir, f"{frame_number:04d}", f"{object_id}.png")
            if os.path.exists(mask_filename):
                mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
                
                # run postprocessing on the mask
                mask = fill_small_holes(mask)
                mask = remove_small_dots(mask)
                mask = grow_shrink(mask)
                mask = border_fix(mask)

                # Resize mask if dimensions don't match
                if mask.shape != (image.shape[0], image.shape[1]):
                    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

                # Find contours for the current mask
                contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                # Choose a distinct color from the palette
                border_color = PALETTE[object_id % len(PALETTE)]  # Cycle through the palette
                thickness = 2  # Thickness of the border

                # Draw contours on the image copy
                cv2.drawContours(image_copy, contours, -1, border_color, thickness)
    return image_copy

# Mask postprocessing - holes
def fill_small_holes(mask):
    max_hole_area = settings["holes"]**2
    filled_mask = mask.copy()
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over contours and fill small holes
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area <= max_hole_area and hierarchy[0][i][3] != -1:  # Check if it's a hole (child contour)
            # Fill the hole by drawing the contour on the original mask
            cv2.drawContours(filled_mask, [contour], -1, 255, thickness=cv2.FILLED)

    return filled_mask

# Mask postprocessing - dots
def remove_small_dots(mask):
    max_dot_area = settings["dots"]**2
    num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
    cleaned_mask = np.zeros_like(mask)

    for label in range(1, num_labels):  # Skip label 0 (background)
        # Get the pixels belonging to this label
        component = (labels == label).astype(np.uint8)

        # Calculate the area of the component
        area = np.sum(component)

        if area > max_dot_area:
            # Keep larger components
            cleaned_mask[labels == label] = 255

    return cleaned_mask

# Mask postprocessing - grow/shrink
def grow_shrink(mask):
    grow_value = settings["grow"]
    kernel = np.ones((abs(grow_value)+1, abs(grow_value)+1), np.uint8)
    if grow_value > 0:
        return cv2.dilate(mask, kernel, iterations=1)
    elif grow_value < 0:
        return cv2.erode(mask, kernel, iterations=1)
    else:
        return mask

def border_fix(mask):
    border_size = settings["border"]
    if border_size == 0:
        return mask
    else: 
        height, width = mask.shape
        y_start = border_size
        y_end = height - border_size
        x_start = border_size
        x_end = width - border_size
        return cv2.copyMakeBorder(mask[y_start:y_end, x_start:x_end], border_size, border_size, border_size, border_size, cv2.BORDER_REPLICATE, value=None)

# Draw points on the current frame
def draw_points(image, frame_number):
    for point in points_list:
        if point.Frame == frame_number:
            point_color = (0, 255, 0) if point.Positive else (255, 0, 0)
            image = cv2.circle(image, (point.X, point.Y), 5, (255,255,0), 2) #yellow outline
            image = cv2.circle(image, (point.X, point.Y), 4, point_color, -1) #filled circle
    return image

def lock_ui():
    return [gr.Button(value="Track Objects", visible=False), gr.Button(value="Cancel", visible=True), gr.Button(value="Undo Last Point", interactive=False), gr.Button(value="Clear Object (frame)", interactive=False), gr.Button(value="Clear Object", interactive=False), gr.Button(value="Clear Tracking Data", interactive=False), gr.Button(value="Clear All", interactive=False), gr.Button(value="Dedupe Masks", interactive=False), gr.File(label="Upload Video or Image File", file_types=['video', '.mkv', 'image'], interactive=False), gr.Tab(label="Matting", visible=False), gr.Tab(label="Export", visible=False)]

def lock_ui_dedupe():
    return [gr.Button(value="Track Objects", interactive=False), gr.Button(value="Cancel", visible=False), gr.Button(value="Undo Last Point", interactive=False), gr.Button(value="Clear Object (frame)", interactive=False), gr.Button(value="Clear Object", interactive=False), gr.Button(value="Clear Tracking Data", interactive=False), gr.Button(value="Clear All", interactive=False), gr.Button(value="Dedupe Masks", interactive=False), gr.File(label="Upload Video or Image File", file_types=['video', '.mkv', 'image'], interactive=False), gr.Tab(label="Matting", visible=False), gr.Tab(label="Export", visible=False)]

def unlock_ui():
    return [gr.Button(value="Track Objects", visible=True, interactive=True), gr.Button(value="Cancel", visible=False), gr.Button(value="Undo Last Point", interactive=True), gr.Button(value="Clear Object (frame)", interactive=True), gr.Button(value="Clear Object", interactive=True), gr.Button(value="Clear Tracking Data", interactive=True), gr.Button(value="Clear All", interactive=True), gr.Button(value="Dedupe Masks", interactive=True), gr.File(label="Upload Video or Image File", file_types=['video', '.mkv', 'image'], interactive=True), gr.Tab(label="Matting", visible=True), gr.Tab(label="Export", visible=True)]


def lock_ui_matting():
    return [gr.Button(value="Run Matting (based on segmentation mask of selected frame)", visible=False), gr.Button(value="Cancel Matting", visible=True), gr.Radio(["Segmentation Mask", "Matting Result"], label="Viewer Output", value="Matting Result", interactive=False), gr.File(label="Upload Video or Image File", file_types=['video', '.mkv', 'image'], interactive=False), gr.Tab(label="Segmentation", visible=False), gr.Tab(label="Export", visible=False)]

def unlock_ui_matting():
    return [gr.Button(value="Run Matting (based on segmentation mask of selected frame)", visible=True), gr.Button(value="Cancel Matting", visible=False), gr.Radio(["Segmentation Mask", "Matting Result"], label="Viewer Output", value="Matting Result", interactive=True), gr.File(label="Upload Video or Image File", file_types=['video', '.mkv', 'image'], interactive=True), gr.Tab(label="Segmentation", visible=True), gr.Tab(label="Export", visible=True)]

def propagate_masks():
    global propagating
    propagating = True
    frame_count = count_frames()
    predictor.reset_state(inference_state)
    replay_points_sequentially()
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=0):
        if not propagating:
            break
        for i, out_obj_id in enumerate(out_obj_ids):
            mask_filename = os.path.join(mask_dir, f"{out_frame_idx:04d}", f"{out_obj_id}.png")
            mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
            mask = (mask * 255).astype(np.uint8) # convert to uint8 before saving to file
            os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
            cv2.imwrite(mask_filename, mask)
        if out_frame_idx % 10 == 0: # update preview every 10 frames
            yield [gr.Slider(minimum=0,maximum=frame_count-1, value=out_frame_idx, step=1, label="Frame Number"), update_image(out_frame_idx)]
    if not propagating:
        print("Tracking cancelled")
        clear_tracking()
    else:
        global propagated 
        propagated= True
    propagating = False
    # measuring memory usage
    #print(torch.cuda.max_memory_allocated(device="cuda") / (1024 ** 3))
    #print(torch.cuda.max_memory_reserved(device="cuda") / (1024 ** 3))
    yield [gr.Slider(minimum=0,maximum=frame_count-1, value=0, step=1, label="Frame Number"), update_image(0)]

def cancel_propagation():
    global propagating
    propagating = False

def cancel_matting():
    global propagating
    propagating = False

def undo_point():
    if points_list:
        point = points_list.pop()
        mask_filename = os.path.join(mask_dir, f"{point.Frame:04d}", f"{point.ObjectID}.png")
        if os.path.exists(mask_filename):
            os.remove(mask_filename)
        clear_tracking() # clear tracking data (if it exists) and replay the points
        save_json(points_list)
    return points_list

# Clear tracking data by deleting all masks then replaying the points
def clear_tracking():
    if os.path.exists(mask_dir):
        shutil.rmtree(mask_dir)
    os.makedirs(mask_dir)
    predictor.reset_state(inference_state)
    clear_cache()
    global propagated
    if propagated:
        gr.Warning("Tracking data cleared.")
    propagated = False
    replay_points_sequentially()
    
def clear_all_points():
    points_list.clear()
    if os.path.exists(mask_dir):
        shutil.rmtree(mask_dir)
    os.makedirs(mask_dir)
    save_json(points_list)
    predictor.reset_state(inference_state)
    clear_cache()
    global propagated
    propagated = False
    return points_list

# Clear points on the current frame for the specified object
def clear_points_obj(frame_number, object_id):
    found = False
    global points_list
    for point in points_list:
        if point.Frame == frame_number and point.ObjectID == object_id:
            found = True
    if found: # only continue if points are found on current frame
        points_list = [point for point in points_list if not (point.Frame == frame_number and point.ObjectID == object_id)]
        mask_filename = os.path.join(mask_dir, f"{frame_number:04d}", f"{object_id}.png")
        if os.path.exists(mask_filename):
            os.remove(mask_filename)
        clear_tracking() #we need to clear the tracking data if removing points from an object
        save_json(points_list)
    return points_list

def clear_all_points_obj(object_id):
    global points_list
    points_list = [point for point in points_list if point.ObjectID != object_id]
    for frame_number in range(count_frames()):
        mask_filename = os.path.join(mask_dir, f"{frame_number:04d}", f"{object_id}.png")
        if os.path.exists(mask_filename):
            os.remove(mask_filename)
    save_json(points_list)
    predictor.remove_object(inference_state, object_id)
    if len(points_list) == 0:
            predictor.reset_state(inference_state)
    return points_list

# Change the color displayed in the interface to indicate the current object
def update_color(object_id):
    color = '#%02x%02x%02x' % PALETTE[object_id % len(PALETTE)] # convert color palette to hex
    return gr.ColorPicker(label="Object Color", value=color, interactive=False, min_width=100)


# set the matting frame slider to the same frame as the segmentation frame slider, and vice versa
def sync_sliders(frame_number): 
    frame_count = count_frames()
    return gr.Slider(minimum=0,maximum=frame_count-1, value=frame_number, step=1, label="Frame Number")

# Function to update the matting preview image based on the slider value
def update_image_mat(slider_value, radio_value):
    frame_filename = os.path.join(frames_dir, f"{slider_value:04d}.png")
    image = cv2.imread(frame_filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if radio_value == "Segmentation Mask":
        if os.path.exists(frame_filename):
            image = draw_masks(image, slider_value)
            #image = draw_contours(image, slider_value)
            #image = draw_points(image, slider_value)
            return image
        else:
            return None
    else:
        object_ids = get_objects()
        if not object_ids:
            return image
        combined_mask = np.zeros_like(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), dtype=np.uint8)  # Initialize combined mask as blank
        for object_id in object_ids:
            mask_filename = os.path.join(matting_dir, f"{slider_value:04d}", f"{object_id}.png")
            if os.path.exists(mask_filename):
                mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
                mask = grow_shrink_matte(mask)
                mask = gamma(mask)
                if mask is not None and np.any(mask):  # Ensure mask is not blank
                    combined_mask = cv2.bitwise_or(combined_mask, mask)
        return combined_mask

# Mask postprocessing - grow/shrink
def grow_shrink_matte(matte):
    grow_value = settings["grow_matte"]
    kernel = np.ones((abs(grow_value)+1, abs(grow_value)+1), np.uint8)
    if grow_value > 0:
        return cv2.dilate(matte, kernel, iterations=1)
    elif grow_value < 0:
        return cv2.erode(matte, kernel, iterations=1)
    else:
        return matte

def gamma(matte):
    gamma_value = settings["gamma"]
    # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    invGamma = 1.0 / gamma_value
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(matte, table)

def resize_image(image):
    max_size = settings["matting_quality"]
    if max_size > 0:
        h, w = image.shape[:2]
        min_side = min(h, w)
        if min_side > max_size:
            scale = max_size / min_side
            new_h = int(h * scale)
            new_w = int(w * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

def restore_image_size(image, original_size):
    original_h, original_w = original_size
    restored_image = cv2.resize(image, (original_h, original_w), interpolation=cv2.INTER_LINEAR)
    return restored_image
      
@torch.inference_mode() 
def run_matting(start_frame):
    clear_cache()
    global propagating
    propagating = True
    frame_count = count_frames()
    object_ids = get_objects()
    total_frame_count = frame_count * len(object_ids)
    progress = tqdm(total=total_frame_count, desc="Matting Progress")

    if os.path.exists(matting_dir):
        shutil.rmtree(matting_dir)
    os.makedirs(matting_dir)

    # get list of image paths
    images = []
    for frame_number in range(frame_count):
        image_filename = os.path.join(frames_dir, f"{frame_number:04d}.png")
        if os.path.exists(image_filename):
            images.append(image_filename)

    with torch.amp.autocast(enabled=False, device_type=device.type, dtype=torch.float16):  # slightly reduces memory consumption but decreases quality. Currently leaving off.
        for object_id in object_ids:
            # load the first-frame mask
            mask_filename = os.path.join(mask_dir, f"{start_frame:04d}", f"{object_id}.png")
            if os.path.exists(mask_filename):
                mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
            else:   
                print("Mask not found for object", object_id)
                gr.Warning("Mask not found for object", object_id)
                continue
            if mask is None or not np.any(mask):  # Ensure mask is not blank
                print("Mask is blank for object", object_id)
                gr.Warning("Mask is blank for object", object_id)
                continue
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            mask = fill_small_holes(mask)
            mask = remove_small_dots(mask)
            original_size = mask.shape[1::-1]
            mask = resize_image(mask) # resize mask to matting quality
            mask = torch.tensor(mask, dtype=torch.float32, device=device)

            # special case if the sequence is only a single frame
            if frame_count == 1:
                frame_path = images[0]
                img = cv2.imread(frame_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = resize_image(img)
                img = torch.tensor(img / 255., dtype=torch.float32, device=device).permute(2, 0, 1)

                output_prob = mat_processor.step(img, mask, objects=[1])
                for i in range(10):
                    output_prob = mat_processor.step(img, first_frame_pred=True)
                    clear_cache()

                mat = mat_processor.output_prob_to_mask(output_prob)
                mat = mat.detach().cpu().numpy()
                mat = (mat * 255).astype(np.uint8)
                mat = restore_image_size(mat, original_size)

                mat_filename = os.path.join(matting_dir, f"0000", f"{object_id}.png")
                os.makedirs(os.path.dirname(mat_filename), exist_ok=True)
                cv2.imwrite(mat_filename, mat)
                progress.update(1)
                yield gr.Slider(minimum=0, maximum=0, value=0, step=1, label="Frame Number")
                continue  # skip to next object

            # forward loop from start frame
            if start_frame < frame_count - 1: # only run this block if we are not starting from the last frame
                for frame_number, frame_path in enumerate(images[start_frame:], start=start_frame): 
                    if not propagating: # checks for the user to cancel
                        break
                    img = cv2.imread(frame_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = resize_image(img) # resize image to matting quality
                    img = torch.tensor(img / 255., dtype=torch.float32, device=device).permute(2, 0, 1)  # Normalize and reorder channels

                    if frame_number == start_frame:
                        output_prob = mat_processor.step(img, mask, objects=[1])      # encode given mask
                        for i in range(10): # warmup by processing first frame 10 times
                            output_prob = mat_processor.step(img, first_frame_pred=True)      # first frame for prediction
                            clear_cache()
                        yield gr.Slider(minimum=0,maximum=frame_count-1, value=frame_number, step=1, label="Frame Number")
                    else:
                        output_prob = mat_processor.step(img)

                    # convert output probabilities to alpha matte
                    mat = mat_processor.output_prob_to_mask(output_prob)
                    mat = mat.detach().cpu().numpy()
                    mat = (mat*255).astype(np.uint8)
                    mat = restore_image_size(mat, original_size)

                    mat_filename = os.path.join(matting_dir, f"{frame_number:04d}", f"{object_id}.png")
                    os.makedirs(os.path.dirname(mat_filename), exist_ok=True)
                    cv2.imwrite(mat_filename, mat)
                    clear_cache()
                    progress.update(1)
                    if frame_number % 10 == 0: # update preview every 10 frames
                        yield gr.Slider(minimum=0,maximum=frame_count-1, value=frame_number, step=1, label="Frame Number")

            # backward loop from start frame
            if start_frame > 0: # only run this block if we are not starting from the first frame
                for frame_number in range(start_frame, -1, -1): 
                    if not propagating: # checks for the user to cancel
                        break
                    frame_path = images[frame_number]
                    img = cv2.imread(frame_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = resize_image(img) # resize image to matting quality
                    img = torch.tensor(img / 255., dtype=torch.float32, device=device).permute(2, 0, 1)  # Normalize and reorder channels

                    if frame_number == start_frame:
                        output_prob = mat_processor.step(img, mask, objects=[1])      # encode given mask
                        for i in range(10): # warmup by processing first frame 10 times
                            output_prob = mat_processor.step(img, first_frame_pred=True)      # first frame for prediction
                            clear_cache()
                    else:
                        output_prob = mat_processor.step(img)

                    # convert output probabilities to alpha matte
                    mat = mat_processor.output_prob_to_mask(output_prob)
                    mat = mat.detach().cpu().numpy()
                    mat = (mat*255).astype(np.uint8)
                    mat = restore_image_size(mat, original_size)

                    mat_filename = os.path.join(matting_dir, f"{frame_number:04d}", f"{object_id}.png")
                    os.makedirs(os.path.dirname(mat_filename), exist_ok=True)
                    cv2.imwrite(mat_filename, mat)
                    clear_cache()
                    progress.update(1)
                    if frame_number % 10 == 0: # update preview every 10 frames
                        yield gr.Slider(minimum=0,maximum=frame_count-1, value=frame_number, step=1, label="Frame Number")
    
    # measuring memory usage
    #print(torch.cuda.max_memory_allocated(device="cuda") / (1024 ** 3))
    #print(torch.cuda.max_memory_reserved(device="cuda") / (1024 ** 3))
    
    progress.close()
    if not propagating: # if cancelled, delete matting dir
        print("Matting Cancelled")
        if os.path.exists(matting_dir):
            shutil.rmtree(matting_dir)
        os.makedirs(matting_dir)
    propagating = False
    clear_cache()
    yield gr.Slider(minimum=0,maximum=frame_count-1, value=frame_number, step=1, label="Frame Number")

def update_export_objects():
    return gr.Dropdown(choices=["All"]+get_objects(), label="Export Object", interactive=True)

def export_image(type, content, object):
    object_ids = get_objects()
    img = None
    mask = None
    frame_path = os.path.join(frames_dir, "0000.png")
    
    # Build dynamic filename based on session settings
    base_filename = build_video_filename()
    image_filename = os.path.join(temp_dir, f"{base_filename}.png")

    if content == "Matting":
        if not os.path.exists(os.path.join(matting_dir, "0000")):
            gr.Warning("No mattes to export. Please \"Run Matting\" from the Matting tab first.")
            return ("No mattes to export. Please \"Run Matting\" from the Matting tab first.", None)
        mask_folder = os.path.join(matting_dir, "0000")
    else:
        if not os.path.exists(os.path.join(mask_dir, "0000")):
            gr.Warning("No masks to export. Please create a mask first.")
            return ("No masks to export. Please create a mask first.", None)
        mask_folder = os.path.join(mask_dir, "0000")
    
    if object == "All":
        for i, object_id in enumerate(object_ids):
            file_path = os.path.join(mask_folder, f"{object_id}.png")
            current_mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if content != "Matting":
                _, current_mask = cv2.threshold(current_mask, 127, 255, cv2.THRESH_BINARY)
                current_mask = fill_small_holes(current_mask)
                current_mask = remove_small_dots(current_mask)
                current_mask = grow_shrink(current_mask)
                current_mask = border_fix(current_mask)
            elif content == "Matting":
                current_mask = gamma(current_mask)
                current_mask = grow_shrink_matte(current_mask)
            # Initialize or accumulate
            if mask is None:
                mask = current_mask
            else:
                mask = cv2.bitwise_or(mask, current_mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        mask_filename = os.path.join(mask_folder, f"{object}.png")
        mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
        if content != "Matting":
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            mask = fill_small_holes(mask)
            mask = remove_small_dots(mask)
            mask = grow_shrink(mask)
            mask = border_fix(current_mask)
        elif content == "Matting":
            mask = gamma(mask)
            mask = grow_shrink_matte(mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    if content == "Segmentation with Edge Smoothing":
        smoothing_model = prepare_smoothing_model("./checkpoints/1x_binary_mask_smooth.pth", device)
        mask = run_smoothing_model(mask, smoothing_model, device)
    if type=="Alpha": 
        img = cv2.imread(frame_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        img[:, :, 3] = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2BGRA)
        img = img * (mask/255)
        img = img.astype(np.uint8)
        #frame = av.VideoFrame.from_ndarray(img, format='bgra')
    elif type=="Greenscreen":
        img = cv2.imread(frame_path)
        img = img * (mask/255)
        green = np.zeros_like(img)
        green[:, :] = [0, 255, 0]
        img = img + (green * (cv2.bitwise_not(mask) / 255))
        img = img.astype(np.uint8)
        #frame = av.VideoFrame.from_ndarray(img, format='bgr24')
    else: # type=="Matte"
        img = mask
        #frame = av.VideoFrame.from_ndarray(img, format='bgr24')
    cv2.imwrite(image_filename, img)
    return (f"Exported image to {image_filename}", gr.DownloadButton(label="💾 Download Exported Image", value=image_filename, visible=True))

def export_video(fps, type, content, object, progress=gr.Progress()):
    frame_count = count_frames()
    total_masks = 0
    object_ids = get_objects()
    object_count = len(object_ids)

    if content == "Matting":
        if not os.path.exists(matting_dir):
            gr.Warning("No mattes to export. Please \"Run Matting\" from the Matting tab first.")
            return ("No mattes to export. Please \"Run Matting\" from the Matting tab first.", None)
        total_masks = sum([len(files) for _, _, files in os.walk(matting_dir)])
        if frame_count*object_count != total_masks:
            gr.Warning("Not all frames have mattes. Please \"Run Matting\" from the Matting tab.")
            return ("Not all frames have mattes. Please \"Run Matting\" from the Matting tab.", None)
    else:
        if not os.path.exists(mask_dir):
            gr.Warning("No masks to export. Please run \"Track Objects\" first.")
            return ("No masks to export. Please run \"Track Objects\" first.", None)
            
        total_masks = sum([len(files) for _, _, files in os.walk(mask_dir)])
        if frame_count*object_count != total_masks:
            gr.Warning("Not all frames have masks. Please run \"Track Objects\".")
            return ("Not all frames have masks. Please run \"Track Objects\".", None)

    images = []
    masks = []
    for frame_number in range(frame_count):
        image_filename = os.path.join(frames_dir, f"{frame_number:04d}.png")
        if os.path.exists(image_filename):
            images.append(image_filename)
    height, width, _ = cv2.imread(images[0]).shape
    
    # Build dynamic filename based on session settings
    base_filename = build_video_filename()
    if type=="Alpha":
        video_filename = os.path.join(temp_dir, f"{base_filename}.mov")
    else: # type=="Matte, Greenscreen"
        video_filename = os.path.join(temp_dir, f"{base_filename}.mp4")

    try:
        fps = float(fps)
    except ValueError:
        gr.Warning("Invalid FPS value. Please enter a number.")
        return ("Invalid FPS value. Please enter a number.", None)
    # convert float framerates to fraction
    if isinstance(fps, float):
        if fps == 29.97:
            fps = Fraction(30000, 1001)
        elif fps == 23.976:
            fps = Fraction(24000, 1001)
        elif fps == 59.94:
            fps = Fraction(60000, 1001)
        else:
            fps = Fraction(fps)
            fps = fps.limit_denominator(0x7fffffff)

    if content == "Segmentation with Edge Smoothing":
        smoothing_model = prepare_smoothing_model("./checkpoints/1x_binary_mask_smooth.pth", device)

    # Prepare the output file with PyAV
    progress(0)
    output = av.open(video_filename, mode='w')
    if type=="Alpha":
        stream = output.add_stream("prores_ks", rate=fps, pix_fmt='yuva444p10le', options={'profile':'4'})
        stream.width = width
        stream.height = height
    else: # type=="Matte, Greenscreen"
        stream = output.add_stream('h264', rate=fps, pix_fmt = 'yuv420p', options={"crf": "10"})
        # For h264, add padding if the resolution is not mod 2
        if width%2 == 1:
            stream.width = width + 1
        else:
            stream.width = width
        if height%2 == 1:
            stream.height = height = 1
        else: 
            stream.height = height


    # read the frames and masks to build an output image
    for frame_number, frame_path in enumerate(images):
        img = None
        mask = None
        if content == "Matting":
            mask_folder = os.path.join(matting_dir, f"{frame_number:04d}")
        else:
            mask_folder = os.path.join(mask_dir, f"{frame_number:04d}")
        if object == "All":
            for i, object_id in enumerate(object_ids):
                file_path = os.path.join(mask_folder, f"{object_id}.png")
                current_mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if content != "Matting":
                    _, current_mask = cv2.threshold(current_mask, 127, 255, cv2.THRESH_BINARY)
                    current_mask = fill_small_holes(current_mask)
                    current_mask = remove_small_dots(current_mask)
                    current_mask = grow_shrink(current_mask)
                    current_mask = border_fix(current_mask)
                elif content == "Matting":
                    current_mask = gamma(current_mask)
                    current_mask = grow_shrink_matte(current_mask)
                # Initialize or accumulate
                if mask is None:
                    mask = current_mask
                else:
                    mask = cv2.bitwise_or(mask, current_mask)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            mask_filename = os.path.join(mask_dir, f"{frame_number:04d}", f"{object}.png")
            mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
            if content != "Matting":
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                mask = fill_small_holes(mask)
                mask = remove_small_dots(mask)
                mask = grow_shrink(mask)
                mask = border_fix(current_mask)
            elif content == "Matting":
                mask = gamma(mask)
                mask = grow_shrink_matte(mask)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if content == "Segmentation with Edge Smoothing":
            mask = run_smoothing_model(mask, smoothing_model, device)
        if type=="Alpha": 
            img = cv2.imread(frame_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            img[:, :, 3] = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2BGRA)
            img = img * (mask/255)
            img = img.astype(np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format='bgra')
        elif type=="Greenscreen":
            img = cv2.imread(frame_path)
            img = img * (mask/255)
            green = np.zeros_like(img)
            green[:, :] = [0, 255, 0]
            img = img + (green * (cv2.bitwise_not(mask) / 255))
            img = img.astype(np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format='bgr24')
        else: # type=="Matte"
            img = mask
            frame = av.VideoFrame.from_ndarray(img, format='bgr24')
        packet = stream.encode(frame)
        output.mux(packet)
        progress(frame_number/frame_count)
    
    # flush
    packet = stream.encode(None)
    output.mux(packet)
    output.close()
    progress(1)
    return (f"Exported video to {video_filename}", gr.DownloadButton(label="💾 Download Exported Video", value=video_filename, visible=True))

# ........................................................................................................................................
# Create the Gradio Blocks interface
# ........................................................................................................................................
with gr.Blocks(title='Sammie-Roto') as demo:

    start_update_check()
    settings = load_settings()
    session = load_session()
    device = setup_cuda()
    predictor = load_model()
    frame_count = count_frames()
    mat_processor = load_matting_model()
    dedupe_min_threshold = settings.get("dedupe_min_threshold")
    save_settings() # save settings in case defaults have not been saved yet

    # resume previous session
    if os.path.exists(temp_dir):
        json_filename = os.path.join(temp_dir, "points.json")
        if os.path.exists(json_filename):
            with open(json_filename, 'r') as f:
                json_data = json.load(f)
                points_list = [DataPoint(*point) for point in json_data]
        if os.path.exists(frames_dir) and os.listdir(frames_dir): # if there are frames
            print("Resuming previous session...")
            inference_state = predictor.init_state(video_path=frames_dir, async_loading_frames=True, offload_video_to_cpu=True)

    # Define the Gradio components
    with gr.Sidebar():
        gr.Markdown("### Input Video / Settings")
        video_input = gr.File(label="Upload Video or Image File", file_types=['video', '.mkv', 'image'])
        model_dropdown = gr.Dropdown(choices=["Auto", "SAM2.1Large (High Quality)", "SAM2.1Base+", "EfficientTAM (Fast)"], value=set_model_dropdown(), label="Segmentation Model", interactive=True)
        matting_quality_dropdown = gr.Dropdown(choices=["480p", "720p", "1080p", "Full"], value=set_matting_quality_dropdown(), label="Matting Max Internal Size", interactive=True)
        cpu_checkbox = gr.Checkbox(label="Force Processing on CPU", value=settings["force_cpu"], interactive=True)
        load_points_btn = gr.UploadButton(label="Load points from file", file_types=['.json'], interactive=True)
        save_points_btn = gr.DownloadButton(label="Save points to file", value=os.path.join(temp_dir, "points.json"), interactive=True)

    with gr.Tab("Segmentation") as segmentation_tab:
        with gr.Accordion(label="Instructions (Click to expand/collapse)", open=False):
            gr.Markdown(
            """
            - Before starting, you need a short video that has been trimmed to a single scene. You can find some sample videos in the "examples" folder. The video can be loaded using the sidebar to the left.
            - You can drag the slider to seek through the frames, and click to add points.
            - You can track multiple different objects at the same time by changing the object id. Note that each additional object will cause tracking to be slower.
            - Press the \"Track Objects\" button to track the mask across all frames of the video.
            - If you add or remove any points after tracking, the tracking data will be cleared, and you must run tracking again.
            - The sliders at the bottom can be used to make adjustments to the masks.
            - Optionally press \"Dedupe Masks\" to suppress edge chatter on duplicated frames, such as with anime/cartoons.
            - When you are satisfied with the result, move to the Export tab at the top to render the video.
            """)
        
        image_viewer = gr.Image(label="Frame Viewer", interactive=False, show_download_button=False, show_label=False)
        frame_slider = gr.Slider(0, frame_count-1, step=1, label="Frame Number")
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    object_id = gr.Number(label="Object ID", value=0, minimum=0, maximum=20, step=1, interactive=True, min_width=100)
                    color_picker = gr.ColorPicker(label="Object Color", value="#800000", interactive=False, min_width=100)
                    point_type = gr.Radio(["+", "-"], label="Point Type", value="+", interactive=True)
            with gr.Column(scale=3):
                with gr.Row():
                    undo_point_btn = gr.Button(value="Undo Last Point")
                    clear_points_obj_btn = gr.Button(value="Clear Object (frame)")
                    clear_all_points_obj_btn = gr.Button(value="Clear Object")
                with gr.Row():
                    propagate_btn = gr.Button(value="Track Objects")
                    cancel_propagate_btn = gr.Button(value="Cancel", visible=False)
                    clear_tracking_btn = gr.Button(value="Clear Tracking Data")
                    clear_all_points_btn = gr.Button(value="Clear All")

        with gr.Row():
            post_holes_slider = gr.Slider(minimum=0, maximum=50, value=set_postprocessing_holes_slider(), step=1, label="Remove Holes")
            post_dots_slider = gr.Slider(minimum=0, maximum=50, value=set_postprocessing_dots_slider(), step=1, label="Remove Dots")
            post_grow_slider = gr.Slider(minimum=-10, maximum=10, value=set_postprocessing_grow_slider(), step=1, label="Shrink/Grow")
            post_border_slider = gr.Slider(minimum=0, maximum=10, value=set_postprocessing_border_slider(), step=1, label="Border Fix")
            with gr.Column():
                show_outlines_checkbox = gr.Checkbox(label="Show Outlines", value=set_show_outlines(), interactive=True)
                dedupe_masks_btn = gr.Button(value="Dedupe Masks")
        with gr.Accordion(label="Point List", open=False):
            point_viewer = gr.Dataframe(
                    headers=["Frame", "ObjectID", "Positive", "X", "Y"],
                    datatype=["number", "number", "number", "number", "number"],
                    value=points_list,
                    col_count=5)
    with gr.Tab("Matting") as matting_tab:
        with gr.Accordion(label="Instructions (Click to expand/collapse)", open=False):
            gr.Markdown(
            """
            - Matting works well for objects with soft or poorly defined edges, such as hair or fur.
            - Matting requires you to first create a mask on at least one frame in the segmentation tab.
            - Set the frame slider below to display the frame that you want to use as the input for the matting model. (The frame must contain a mask)
            - Click the \"Run Matting\" button to run matting across the entire video. You can sometimes get better results by trying from a different starting frame.
            - The sliders at the bottom can be used to make adjustments to the matting result.
            - If you are satisfied with the result, move to the Export tab to render the video.
            """)
        image_viewer_mat = gr.Image(label="Frame Viewer", interactive=False, show_download_button=False, show_label=False)
        frame_slider_mat = gr.Slider(0, frame_count-1, step=1, label="Frame Number")
        viewer_output_radio = gr.Radio(["Segmentation Mask", "Matting Result"], label="Viewer Output", value="Segmentation Mask", interactive=True)
        matting_btn = gr.Button(value="Run Matting (based on segmentation mask of selected frame)")
        cancel_matting_btn = gr.Button(value="Cancel Matting", visible=False)
        with gr.Row():
            post_gamma_slider = gr.Slider(minimum=0.01, maximum=10, value=set_postprocessing_gamma_slider(), step=0.01, label="Gamma")
            post_grow_matte_slider = gr.Slider(minimum=-10, maximum=10, value=set_postprocessing_grow_matte_slider(), step=1, label="Shrink/Grow")


    with gr.Tab("Export") as export_tab:
        with gr.Accordion(label="Instructions (Click to expand/collapse)", open=False):
            gr.Markdown(
            """
            - Before exporting, make sure you have run \"Track Objects\" under the segmentation page in order to generate masks for every frame, or if you want to export the matting result, make sure you have \"Run Matting\" under the matting page.
            - Three export types are available: "Matte" exports a black and white matte. "Alpha" exports the masked objects with an alpha channel. "Greenscreen" exports the masked objects with a solid green background. "Matte" and "Greenscreen" will export a high quality MP4 file, while "Alpha" will export a ProRes file.
            - Export content options include "Segmentation Mask", "Segmentation with Edge Smoothing", and "Matting". The edge smoothing option will run the segmentation masks through an antialiasing model to smooth out the edges.
            - The postprocessing options at the bottom of the segmentation page or the matting page will affect the result, so make sure they are set correctly before exporting.
            - Export object options include "All" to export all objects combined into a single video, or you can select a specific object ID to export.
            - In the file naming options, you can choose to add the SammieRoto suffix, object ID, export type and date & time to the filename. The preview filename will update automatically based on your selections.
            - If you want to give the file a custom name, you can do so after exporting and then clicking the download button.
            """)        
        export_fps = gr.Dropdown(choices=[23.976, 24, 29.97, 30], value=str(settings['export_fps']), label="FPS", allow_custom_value=True, interactive=True)
        export_type = gr.Dropdown(choices=["Matte", "Alpha", "Greenscreen"], value=set_export_type_dropdown(), label="Export Type", interactive=True)
        export_content = gr.Dropdown(choices=["Segmentation Mask", "Segmentation with Edge Smoothing", "Matting"], value=set_export_content_dropdown(), label="Export Content", interactive=True)
        export_object = gr.Dropdown(choices=["All"]+get_objects(), label="Export Object", interactive=True)
        
        # Name section
        with gr.Accordion(label="File Naming Options", open=True):
            preview_filename = gr.Textbox(
                value=build_video_filename() if session else "No file uploaded",
                label="Preview Filename",
                interactive=False
            )
            with gr.Row():
                name_roto_checkbox = gr.Checkbox(
                    label="Add SammieRoto", 
                    value=session.get("name_roto", True), 
                    interactive=True
                )
                name_type_checkbox = gr.Checkbox(
                    label="Add export type", 
                    value=session.get("name_type", False), 
                    interactive=True
                )
            with gr.Row():
                name_object_checkbox = gr.Checkbox(
                    label="Add layer/object ID", 
                    value=session.get("name_object", False), 
                    interactive=True
                )
                name_date_time_checkbox = gr.Checkbox(
                    label="Add date & time", 
                    value=session.get("name_date_time", False), 
                    interactive=True
                )
        
        with gr.Row():
            export_btn = gr.Button(value="Export Video")
            export_img_btn = gr.Button(value="Export Image")
        export_status = gr.Textbox(value="", label="Export Status")
        export_download = gr.DownloadButton(label="💾 Download Exported Video", visible=False)
    
    # Define the event listeners
    video_input.upload(process_and_enable_slider, inputs=video_input, outputs=[frame_slider, frame_slider_mat, export_fps]).then(clear_all_points, outputs=point_viewer).then(update_image, inputs=frame_slider, outputs=image_viewer).then(reset_postprocessing, outputs=[post_holes_slider, post_dots_slider, post_grow_slider, post_border_slider, show_outlines_checkbox, post_gamma_slider, post_grow_matte_slider]).then(lambda: build_video_filename(), outputs=preview_filename)
    model_dropdown.input(change_settings, inputs=[model_dropdown, matting_quality_dropdown, cpu_checkbox])
    matting_quality_dropdown.input(change_settings, inputs=[model_dropdown, matting_quality_dropdown, cpu_checkbox])
    cpu_checkbox.input(change_settings, inputs=[model_dropdown, matting_quality_dropdown, cpu_checkbox])
    load_points_btn.upload(load_points, inputs=load_points_btn, outputs=point_viewer).then(update_image, inputs=frame_slider, outputs=image_viewer)
    save_points_btn.click(save_points)
    post_holes_slider.input(change_postprocessing, inputs=[post_holes_slider, post_dots_slider, post_grow_slider, post_border_slider,show_outlines_checkbox, post_gamma_slider, post_grow_matte_slider]).then(update_image, inputs=frame_slider, outputs=image_viewer, show_progress='hidden')
    post_dots_slider.input(change_postprocessing, inputs=[post_holes_slider, post_dots_slider, post_grow_slider, post_border_slider, show_outlines_checkbox, post_gamma_slider, post_grow_matte_slider]).then(update_image, inputs=frame_slider, outputs=image_viewer, show_progress='hidden')
    post_grow_slider.input(change_postprocessing, inputs=[post_holes_slider, post_dots_slider, post_grow_slider, post_border_slider, show_outlines_checkbox, post_gamma_slider, post_grow_matte_slider]).then(update_image, inputs=frame_slider, outputs=image_viewer, show_progress='hidden')
    post_border_slider.input(change_postprocessing, inputs=[post_holes_slider, post_dots_slider, post_grow_slider, post_border_slider, show_outlines_checkbox, post_gamma_slider, post_grow_matte_slider]).then(update_image, inputs=frame_slider, outputs=image_viewer, show_progress='hidden')
    show_outlines_checkbox.change(change_postprocessing, inputs=[post_holes_slider, post_dots_slider, post_grow_slider, post_border_slider, show_outlines_checkbox, post_gamma_slider, post_grow_matte_slider]).then(update_image, inputs=frame_slider, outputs=image_viewer, show_progress='hidden')
    post_gamma_slider.input(change_postprocessing, inputs=[post_holes_slider, post_dots_slider, post_grow_slider, post_border_slider, show_outlines_checkbox, post_gamma_slider, post_grow_matte_slider]).then(update_image_mat, inputs=[frame_slider_mat, viewer_output_radio], outputs=image_viewer_mat, show_progress='hidden')
    post_grow_matte_slider.input(change_postprocessing, inputs=[post_holes_slider, post_dots_slider, post_grow_slider, post_border_slider, show_outlines_checkbox, post_gamma_slider, post_grow_matte_slider]).then(update_image_mat, inputs=[frame_slider_mat, viewer_output_radio], outputs=image_viewer_mat, show_progress='hidden')
    frame_slider.change(update_image, inputs=frame_slider, outputs=image_viewer, show_progress='hidden')
    image_viewer.select(add_point, inputs=[frame_slider, object_id, point_type], outputs=point_viewer, show_progress='hidden').then(update_image, inputs=frame_slider, outputs=image_viewer, show_progress='hidden')
    object_id.change(update_color, inputs=object_id, outputs=color_picker, show_progress='hidden')
    undo_point_btn.click(undo_point, outputs=point_viewer).then(update_image, inputs=frame_slider, outputs=image_viewer)
    clear_tracking_btn.click(clear_tracking).then(update_image, inputs=frame_slider, outputs=image_viewer)
    clear_all_points_btn.click(clear_all_points, outputs=point_viewer).then(update_image, inputs=frame_slider, outputs=image_viewer)
    clear_points_obj_btn.click(clear_points_obj, inputs=[frame_slider, object_id], outputs=point_viewer).then(update_image, inputs=frame_slider, outputs=image_viewer)
    clear_all_points_obj_btn.click(clear_all_points_obj, inputs=object_id, outputs=point_viewer).then(update_image, inputs=frame_slider, outputs=image_viewer)
    dedupe_masks_btn.click(lock_ui_dedupe, outputs=[propagate_btn, cancel_propagate_btn, undo_point_btn, clear_points_obj_btn, clear_all_points_obj_btn, clear_tracking_btn, clear_all_points_btn, dedupe_masks_btn, video_input, matting_tab, export_tab]).then(lambda: replace_similar_matte_frames(dedupe_min_threshold)).then(unlock_ui, outputs=[propagate_btn, cancel_propagate_btn, undo_point_btn, clear_points_obj_btn, clear_all_points_obj_btn, clear_tracking_btn, clear_all_points_btn, dedupe_masks_btn, video_input, matting_tab, export_tab]).then(update_image, inputs=frame_slider, outputs=image_viewer)
    propagate_btn.click(lock_ui, outputs=[propagate_btn, cancel_propagate_btn, undo_point_btn, clear_points_obj_btn, clear_all_points_obj_btn, clear_tracking_btn, clear_all_points_btn, dedupe_masks_btn, video_input, matting_tab, export_tab]).then(propagate_masks, outputs=[frame_slider, image_viewer]).then(unlock_ui, outputs=[propagate_btn, cancel_propagate_btn, undo_point_btn, clear_points_obj_btn, clear_all_points_obj_btn, clear_tracking_btn, clear_all_points_btn, dedupe_masks_btn, video_input, matting_tab, export_tab])
    cancel_propagate_btn.click(cancel_propagation)
    point_viewer.select(change_slider, outputs=[frame_slider, object_id, color_picker], show_progress='hidden')
    export_type.input(change_export_settings, inputs=[export_type, export_content, export_object]).then(build_video_filename, outputs=preview_filename)
    export_object.input(change_export_settings, inputs=[export_type, export_content, export_object]).then(build_video_filename, outputs=preview_filename)
    export_content.input(change_export_settings, inputs=[export_type, export_content, export_object])
    export_btn.click(export_video, inputs=[export_fps, export_type, export_content, export_object], outputs=[export_status, export_download])
    export_img_btn.click(export_image, inputs=[export_type, export_content, export_object], outputs=[export_status, export_download])
    export_tab.select(update_export_objects, outputs=export_object)
    segmentation_tab.select(sync_sliders, inputs=[frame_slider_mat], outputs=[frame_slider])
    matting_tab.select(sync_sliders, inputs=[frame_slider], outputs=[frame_slider_mat]).then(update_image_mat, inputs=[frame_slider_mat, viewer_output_radio], outputs=image_viewer_mat, show_progress='hidden')
    frame_slider_mat.change(update_image_mat, inputs=[frame_slider_mat, viewer_output_radio], outputs=image_viewer_mat, show_progress='hidden')
    viewer_output_radio.change(update_image_mat, inputs=[frame_slider_mat, viewer_output_radio], outputs=image_viewer_mat, show_progress='hidden')
    matting_btn.click(lock_ui_matting, outputs=[matting_btn, cancel_matting_btn, viewer_output_radio, video_input, segmentation_tab, export_tab]).then(run_matting, inputs=frame_slider_mat, outputs=frame_slider_mat).then(unlock_ui_matting, outputs=[matting_btn, cancel_matting_btn, viewer_output_radio, video_input, segmentation_tab, export_tab])
    cancel_matting_btn.click(cancel_matting)

    name_roto_checkbox.change(update_name_roto, inputs=name_roto_checkbox, outputs=preview_filename)
    name_type_checkbox.change(update_name_type, inputs=name_type_checkbox, outputs=preview_filename)
    name_object_checkbox.change(update_name_object, inputs=name_object_checkbox, outputs=preview_filename)
    name_date_time_checkbox.change(update_name_date_time, inputs=name_date_time_checkbox, outputs=preview_filename)

    # when the app loads, update the image
    demo.load(fn=update_image, inputs=frame_slider, outputs=image_viewer)

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch(show_error=True, inbrowser=True, show_api=False, debug=False)