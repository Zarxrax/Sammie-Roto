# Import necessary libraries
import gradio as gr
import cv2
import os
import numpy as np
import shutil
import json
import av 
import torch
from collections import namedtuple
from fractions import Fraction
from sammie.smooth import run_smoothing_model, prepare_smoothing_model

# .........................................................................................
# Global variables
# .........................................................................................
temp_dir = "temp"
frames_dir = os.path.join(temp_dir, "frames")
mask_dir = os.path.join(temp_dir, "masks")
settings = None
edge_smoothing = False
device = None
propagated = False
inference_state = None
predictor = None
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
    print("Sammie-Roto version 1.0")
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
        # use bfloat16
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        print("CUDA Compute Capability: ", torch.cuda.get_device_capability())
        # turn on tfloat32 for Ampere GPUs
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    return device

# Load the model and predictor / this can only be called once because the imports will interfere with each other
def load_model():
    if device.type == "cuda":
        torch.cuda.empty_cache() #clean up GPU memory in case models get loaded multiple times
    model_selection = settings.get("segmentation_model")
    if (model_selection == "auto" and device.type == "cpu") or model_selection == "efficient":
        from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor
        print("Using EfficientTAM model")
        checkpoint = "./checkpoints/efficienttam_s_512x512.pt"
        model_cfg = "../configs/efficienttam_s_512x512.yaml"
        return build_efficienttam_video_predictor(model_cfg, checkpoint, device=device)
    elif (model_selection == "auto" and torch.cuda.get_device_properties(0).major < 8) or model_selection == "sam2base":
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

# Save settings if the user changes the settings
def change_settings(model_dropdown, cpu_checkbox):
    if model_dropdown == "Auto":
        settings["segmentation_model"] = "auto"
    elif model_dropdown == "SAM2.1Large (High Quality)":
        settings["segmentation_model"] = "sam2large"
    elif model_dropdown == "SAM2.1Base+":
        settings["segmentation_model"] = "sam2base"
    elif model_dropdown == "EfficientTAM (Fast)":
        settings["segmentation_model"] = "efficient"
    if cpu_checkbox:
        settings["force_cpu"] = True
    else:
        settings["force_cpu"] = False
    save_settings()
    gr.Info("You must restart the application for changes to take effect.")

# Save settings if user changes the postprocessing settings
def change_postprocessing(post_holes_slider, post_dots_slider, post_grow_slider, show_outlines_checkbox):
    settings["holes"] = post_holes_slider
    settings["dots"] = post_dots_slider
    settings["grow"] = post_grow_slider
    settings["show_outlines"] = show_outlines_checkbox
    save_settings()

def reset_postprocessing():
    settings["holes"] = 0
    settings["dots"] = 0
    settings["grow"] = 0
    settings["show_outlines"] = True
    save_settings()
    return [gr.Slider(minimum=0, maximum=50, value=0, step=1, label="Remove Holes"), gr.Slider(minimum=0, maximum=50, value=0, step=1, label="Remove Dots"), gr.Slider(minimum=-10, maximum=10, value=0, step=1, label="Shrink/Grow"), gr.Checkbox(label="Show Outlines", value=True, interactive=True)]

# Load settings from json file
def load_settings():
    default_settings = {
    "segmentation_model": "auto",
    "force_cpu": False,
    "export_fps": 24,
    "holes": 0,
    "dots": 0,
    "grow": 0,
    "show_outlines": True
    }
    try:
        with open("settings.json", 'r') as file:
            settings = json.load(file)
            return settings
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

def set_postprocessing_holes_slider():
    return settings["holes"]

def set_postprocessing_dots_slider():
    return settings["dots"]

def set_postprocessing_grow_slider():
    return settings["grow"]

def set_show_outlines():
    return settings["show_outlines"]

# Function to process the video and save frames as PNGs
def process_video(video_file, progress=gr.Progress()):
    global inference_state
    inference_state = None # empty out the inference state if its populated

    # Create temp directories to save the frames, delete if already exists
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(frames_dir)
    os.makedirs(mask_dir)
    
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
    frame_count = process_video(video_file)
    return [gr.Slider(0,frame_count-1, step=1, label="Frame Number"), gr.Dropdown(choices=[23.976, 24, 29.97, 30], value=str(settings['export_fps']), label="FPS", allow_custom_value=True, interactive=True)]

# Function to modify the value of the frame slider when clicking on the dataframe, also update the displayed object color
def change_slider(event_data: gr.SelectData):
    color = '#%02x%02x%02x' % PALETTE[event_data.row_value[1] % len(PALETTE)] # convert color palette to hex
    return event_data.row_value[0], gr.ColorPicker(label="Object Color", value=color, interactive=False)
        
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
        image = cv2.imread(frame_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # format the input points and labels and run the prediction
        filtered_points = [(point.X, point.Y, point.Positive) for point in points_list if point.Frame == frame_number and point.ObjectID == object_id]
        if filtered_points:
            input_points = np.array([(x, y) for x, y, _ in filtered_points], dtype=np.float32)
            input_labels = np.array([positive for _, _, positive in filtered_points], dtype=np.int32)
        else:
            return

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

# Draw points on the current frame
def draw_points(image, frame_number):
    for point in points_list:
        if point.Frame == frame_number:
            point_color = (0, 255, 0) if point.Positive else (255, 0, 0)
            image = cv2.circle(image, (point.X, point.Y), 5, (255,255,0), 2) #yellow outline
            image = cv2.circle(image, (point.X, point.Y), 4, point_color, -1) #filled circle
    return image

def propagate_masks(progress=gr.Progress()):
    progress(0)
    frame_count = count_frames()
    predictor.reset_state(inference_state)
    replay_points_sequentially()
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=0):
        for i, out_obj_id in enumerate(out_obj_ids):
            mask_filename = os.path.join(mask_dir, f"{out_frame_idx:04d}", f"{out_obj_id}.png")
            mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
            mask = (mask * 255).astype(np.uint8) # convert to uint8 before saving to file
            os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
            cv2.imwrite(mask_filename, mask)
        progress(out_frame_idx/frame_count)
        #yield update_image(out_frame_idx)
    global propagated 
    propagated= True
    progress(1)

def undo_point():
    if points_list:
        point = points_list.pop()
        mask_filename = os.path.join(mask_dir, f"{point.Frame:04d}", f"{point.ObjectID}.png")
        if os.path.exists(mask_filename):
            os.remove(mask_filename)
        save_json(points_list)
        if propagated:
            clear_tracking() #we need to clear the tracking data if removing points from an object
        else:
            predictor.clear_all_prompts_in_frame(inference_state, point.Frame, point.ObjectID)
            segment_image(point.Frame, point.ObjectID) # redraw the masks in case any other points are on the frame        
            if len(points_list) == 0: # if we removed the last point, just reset the state
                predictor.reset_state(inference_state)
    return points_list

# Clear tracking data by deleting all masks then replaying the points
def clear_tracking():
    if os.path.exists(mask_dir):
        shutil.rmtree(mask_dir)
    os.makedirs(mask_dir)
    predictor.reset_state(inference_state)
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
        if len(points_list) == 0:
            predictor.reset_state(inference_state)
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
    return gr.ColorPicker(label="Color", value=color, interactive=False)

def change_smoothing(smoothing_checkbox):
    global edge_smoothing
    edge_smoothing = smoothing_checkbox

def update_export_objects():
    return gr.Dropdown(choices=["All"]+get_objects(), label="Export Object", interactive=True)

def export_video(fps, type, object, progress=gr.Progress()):
    frame_count = count_frames()
    total_masks = sum([len(files) for root, dirs, files in os.walk(mask_dir)])
    object_ids = get_objects()
    object_count = len(object_ids)

    if not os.path.exists(mask_dir):
        gr.Warning("No masks to export. Please run \"Track Objects\" first.")
        return "No masks to export. Please run \"Track Objects\" first."
    if frame_count*object_count != total_masks:
        gr.Warning("Not all frames have masks. Please run \"Track Objects\".")
        return "Not all frames have masks. Please run \"Track Objects\"."

    images = []
    masks = []
    for frame_number in range(frame_count):
        image_filename = os.path.join(frames_dir, f"{frame_number:04d}.png")
        if os.path.exists(image_filename):
            images.append(image_filename)
    height, width, _ = cv2.imread(images[0]).shape
    if type=="Alpha":
        video_filename = os.path.join(temp_dir, "output.mov")
    else: # type=="Matte"
        video_filename = os.path.join(temp_dir, "output.mp4")

    try:
        fps = float(fps)
    except ValueError:
        gr.Warning("Invalid FPS value. Please enter a number.")
        return "Invalid FPS value. Please enter a number."
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

    if edge_smoothing:
        smoothing_model = prepare_smoothing_model("./checkpoints/1x_binary_mask_smooth.pth", device)

    # Prepare the output file with PyAV
    progress(0)
    output = av.open(video_filename, mode='w')
    if type=="Alpha":
        stream = output.add_stream("prores_ks", rate=fps, pix_fmt='yuva444p10le', options={'profile':'4'})
    else: # type=="Matte, Greenscreen"
        stream = output.add_stream('h264', rate=fps, pix_fmt = 'yuv420p', options={"crf": "10"})
    stream.width = width
    stream.height = height
    

    # read the frames and masks to build an output image
    for frame_number, frame_path in enumerate(images):
        img = None
        mask = None
        mask_folder = os.path.join(mask_dir, f"{frame_number:04d}")
        if object == "All":
            masks = [os.path.join(mask_folder, f"{object_id}.png") for object_id in object_ids]
            mask = cv2.imread(masks[0], cv2.IMREAD_GRAYSCALE)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            mask = fill_small_holes(mask)
            mask = remove_small_dots(mask)
            for file_path in masks[1:]:
                current_mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                _, current_mask = cv2.threshold(current_mask, 127, 255, cv2.THRESH_BINARY)
                current_mask = fill_small_holes(current_mask)
                current_mask = remove_small_dots(current_mask)
                mask = cv2.bitwise_or(mask, current_mask)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            mask_filename = os.path.join(mask_dir, f"{frame_number:04d}", f"{object}.png")
            mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            mask = fill_small_holes(mask)
            mask = remove_small_dots(mask)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if edge_smoothing:
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

    settings = load_settings()
    device = setup_cuda()
    predictor = load_model()
    frame_count = max(count_frames()-1,0)

    # resume previous session
    if os.path.exists(temp_dir):
        json_filename = os.path.join(temp_dir, "points.json")
        if os.path.exists(json_filename):
            with open(json_filename, 'r') as f:
                json_data = json.load(f)
                points_list = [DataPoint(*point) for point in json_data]
        if os.path.exists(frames_dir):
            print("Resuming previous session...")
            inference_state = predictor.init_state(video_path=frames_dir, async_loading_frames=True, offload_video_to_cpu=True)
            clear_tracking() # remove any tracking data from previous session and replay the points

    # Define the Gradio components
    with gr.Tab("Segmentation"):
        with gr.Accordion(label="Instructions (Click to expand/collapse)", open=False):
            gr.Markdown(
            """
            - Before starting, you need a short video that has been trimmed to a single scene. Load a video file, then the frames will be extracted, and the video will be loaded into the viewer.
            - You can drag the slider to seek through the frames, and click to add points.
            - You can track multiple different objects at the same time by changing the object id. Note that each additional object will cause tracking to be slower.
            - Press the \"Track Objects\" button to track the mask across all frames of the video.
            - If you add or remove any points after tracking, the tracking data will be cleared, and you must run tracking again.
            - The sliders at the bottom can be used to make adjustments to the masks.
            - When you are satisfied with the result, move to the Export tab at the top to render the video.
            """)
        with gr.Accordion(label="Input Video / Settings", open=False):
            with gr.Row():
                video_input = gr.File(label="Upload Video File", file_types=['video', '.mkv'])
                with gr.Column():
                    model_dropdown = gr.Dropdown(choices=["Auto", "SAM2.1Large (High Quality)", "SAM2.1Base+", "EfficientTAM (Fast)"], value=set_model_dropdown(), label="Segmentation Model", interactive=True)
                    cpu_checkbox = gr.Checkbox(label="Force Processing on CPU", value=settings["force_cpu"], interactive=True)
        image_viewer = gr.Image(label="Frame Viewer", interactive=False, show_download_button=False, show_label=False)
        frame_slider = gr.Slider(0, frame_count, step=1, label="Frame Number")
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    object_id = gr.Number(label="Object ID", value=0, minimum=0, maximum=20, step=1, interactive=True, scale=0)
                    color_picker = gr.ColorPicker(label="Object Color", value="#800000", interactive=False, scale=0)
                    point_type = gr.Radio(["+", "-"], label="Point Type", value="+", interactive=True)
            with gr.Column(scale=3):
                with gr.Row():
                    undo_point_btn = gr.Button(value="Undo Last Point")
                    clear_points_obj_btn = gr.Button(value="Clear Object (frame)")
                    clear_all_points_obj_btn = gr.Button(value="Clear Object")
                with gr.Row():
                    propagate_btn = gr.Button(value="Track Objects")
                    clear_tracking_btn = gr.Button(value="Clear Tracking Data")
                    clear_all_points_btn = gr.Button(value="Clear All")
        with gr.Row():
            post_holes_slider = gr.Slider(minimum=0, maximum=50, value=set_postprocessing_holes_slider(), step=1, label="Remove Holes")
            post_dots_slider = gr.Slider(minimum=0, maximum=50, value=set_postprocessing_dots_slider(), step=1, label="Remove Dots")
            post_grow_slider = gr.Slider(minimum=-10, maximum=10, value=set_postprocessing_grow_slider(), step=1, label="Shrink/Grow")
            show_outlines_checkbox = gr.Checkbox(label="Show Outlines", value=set_show_outlines(), interactive=True)
        with gr.Accordion(label="Point List", open=False):
            point_viewer = gr.List(
                    headers=["Frame", "ObjectID", "Positive", "X", "Y"],
                    datatype=["number", "number", "number", "number", "number"],
                    value=points_list,
                    col_count=5)
    with gr.Tab("Export") as export_tab:
        with gr.Accordion(label="Instructions (Click to expand/collapse)", open=False):
            gr.Markdown(
            """
            - Before exporting, make sure you have run \"Track Objects\" under the segmentation page in order to generate masks for every frame.
            - Select the "Matte" export type to export a black and white matte as a high quality MP4 file.
            - Select the "Alpha" export type to export the masked objects with an alpha channel as a ProRes file.
            - Select the "Greenscreen" export type to export the masked objects with a solid green background as a high quality MP4 file.
            - "Smooth Edges" will run the masks through an antialiasing model to smooth out the edges.
            - The postprocessing options at the bottom of the segmentation page will affect the result, so make sure they are set correctly before exporting.
            """)
        export_fps = gr.Dropdown(choices=[23.976, 24, 29.97, 30], value=str(settings['export_fps']), label="FPS", allow_custom_value=True, interactive=True)
        export_type = gr.Dropdown(choices=["Matte", "Alpha", "Greenscreen"], label="Export Type", interactive=True)
        export_smooth = gr.Checkbox(label="Smooth Edges", value=False, interactive=True)
        export_object = gr.Dropdown(choices=["All"]+get_objects(), label="Export Object", interactive=True)
        export_btn = gr.Button(value="Export Video")
        export_status = gr.Textbox(value="", label="Export Status")
        export_download = gr.DownloadButton(label="💾 Download Exported Video", visible=False)
    
    # Define the event listeners
    video_input.upload(process_and_enable_slider, inputs=video_input, outputs=[frame_slider, export_fps]).then(clear_all_points, outputs=point_viewer).then(update_image, inputs=frame_slider, outputs=image_viewer).then(reset_postprocessing, outputs=[post_holes_slider, post_dots_slider, post_grow_slider, show_outlines_checkbox])
    model_dropdown.input(change_settings, inputs=[model_dropdown, cpu_checkbox])
    cpu_checkbox.input(change_settings, inputs=[model_dropdown, cpu_checkbox])
    post_holes_slider.input(change_postprocessing, inputs=[post_holes_slider, post_dots_slider, post_grow_slider,show_outlines_checkbox]).then(update_image, inputs=frame_slider, outputs=image_viewer, show_progress='hidden')
    post_dots_slider.input(change_postprocessing, inputs=[post_holes_slider, post_dots_slider, post_grow_slider, show_outlines_checkbox]).then(update_image, inputs=frame_slider, outputs=image_viewer, show_progress='hidden')
    post_grow_slider.input(change_postprocessing, inputs=[post_holes_slider, post_dots_slider, post_grow_slider, show_outlines_checkbox]).then(update_image, inputs=frame_slider, outputs=image_viewer, show_progress='hidden')
    show_outlines_checkbox.change(change_postprocessing, inputs=[post_holes_slider, post_dots_slider, post_grow_slider, show_outlines_checkbox]).then(update_image, inputs=frame_slider, outputs=image_viewer, show_progress='hidden')
    frame_slider.change(update_image, inputs=frame_slider, outputs=image_viewer, show_progress='hidden')
    image_viewer.select(add_point, inputs=[frame_slider, object_id, point_type], outputs=point_viewer, show_progress='hidden').then(update_image, inputs=frame_slider, outputs=image_viewer, show_progress='hidden')
    object_id.change(update_color, inputs=object_id, outputs=color_picker, show_progress='hidden')
    undo_point_btn.click(undo_point, outputs=point_viewer).then(update_image, inputs=frame_slider, outputs=image_viewer)
    clear_tracking_btn.click(clear_tracking).then(update_image, inputs=frame_slider, outputs=image_viewer)
    clear_all_points_btn.click(clear_all_points, outputs=point_viewer).then(update_image, inputs=frame_slider, outputs=image_viewer)
    clear_points_obj_btn.click(clear_points_obj, inputs=[frame_slider, object_id], outputs=point_viewer).then(update_image, inputs=frame_slider, outputs=image_viewer)
    clear_all_points_obj_btn.click(clear_all_points_obj, inputs=object_id, outputs=point_viewer).then(update_image, inputs=frame_slider, outputs=image_viewer)
    propagate_btn.click(propagate_masks, outputs=[frame_slider]).then(update_image, inputs=frame_slider, outputs=image_viewer)
    point_viewer.select(change_slider, outputs=[frame_slider, color_picker], show_progress='hidden')
    export_btn.click(export_video, inputs=[export_fps, export_type, export_object], outputs=[export_status, export_download])
    export_smooth.change(change_smoothing, inputs=[export_smooth])
    export_tab.select(update_export_objects, outputs=export_object)

    # when the app loads, update the image
    demo.load(fn=update_image, inputs=frame_slider, outputs=image_viewer)

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch(show_error=True, inbrowser=True, show_api=False, debug=True)