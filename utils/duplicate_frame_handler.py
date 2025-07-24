import cv2
import numpy as np
import os
import shutil
import gradio as gr
from .progress_bar import progress_bar

min_similarity_threshold = 0.8 # The compared matte (alpha) frames need to be at least this similar compared to the base matte alpha frame
max_similarity_threshold = 0.98 # The compared matte (alpha) frame is similar to the point where it doesn't have to be processed / replaced

# Use ORB comparison from opencv to compare two input frames/images for similarity
def orb_comparison(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1, des2)
    similarity_score = len(matches) / max(len(kp1), len(kp2))
    return similarity_score

# To compare without other elements or the background on the frame affecting the comparison, the mask luma matte gets applied to the frame
def generate_matted_frame(frame_path, mask_dir, frame_number):
    frame = cv2.imread(frame_path)
    frame_mask_dir = os.path.join(mask_dir, frame_number)
    if not os.path.exists(frame_mask_dir):
        # Warning could not find masks
        return None
    
    # Create empty mask image
    mask_image = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    # Combine all masks from mask folder into one mask image for comparison
    for matte in os.listdir(frame_mask_dir):
        matte_path = os.path.join(frame_mask_dir, matte)
        matte_image = cv2.imread(matte_path, cv2.IMREAD_GRAYSCALE)
        mask_image = cv2.bitwise_or(mask_image, matte_image)
        
    # Use the combined mask image as the overall mask luma matte
    result_frame = cv2.bitwise_and(frame, frame, mask=mask_image)
    return result_frame

# Replace the masks on disc with a specific "similar frames" list
def replace_files_similar_mattes(mask_dir, similar_frames):
    last_mask_dir = os.path.join(mask_dir, similar_frames[-1])
    file_list = os.listdir(last_mask_dir)
    for i, frame in enumerate(similar_frames):
        if i == len(similar_frames) - 1:  # Skip the last sourcing frame
            break
        for file in file_list:
            file_path = os.path.join(last_mask_dir, file)
            replace_mask_dir = os.path.join(mask_dir, frame, file)
            shutil.copy(file_path, replace_mask_dir)

# Main function to replace similar (matted) frames with one single matte frame
def replace_similar_matte_frames():
    # Resolve absolute path of file back to project root folder
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(utils_dir, ".."))
    
    frames_dir = os.path.join(project_root, "temp", "frames")
    mask_dir = os.path.join(project_root, "temp", "masks")

    frame_numbers = []

    # Check if the frames directory exists
    if not os.path.exists(frames_dir):
        gr.Warning("Could not find frames to dedupe.\nPlease upload a video first.", duration=5)
        return
    # Get the list of propagated frame numbers
    for filename in os.listdir(frames_dir):
        if filename.endswith(".png"):
            frame_numbers.append(os.path.splitext(filename)[0])
            
    # Check if the masks directory has the same amount of masks as the amount of frames in the video file
    num_of_masks = len(os.listdir(mask_dir))
    frames_amount = len(frame_numbers)
    if num_of_masks != frames_amount:
        gr.Warning("Mismatch between frames and masks.\nPlease fully track objects first.", duration=10)
        return

    frame_index = 0 # Keeps track of the current "base" frame for comparisons
    deduped_frames_amount = 0 # Keep track of how many frames/masks have been replaced/deduped

    # Initialize the base frame
    start_base_frame_path = os.path.join(frames_dir, frame_numbers[frame_index] + ".png")
    base_frame = generate_matted_frame(start_base_frame_path, mask_dir, frame_numbers[frame_index]) # Frame used for ORB comparison

    gr.Info("Deduping matte frames...", duration=3)
    print("Deduping matte frames...")

    while True:
        progress = frame_index / frames_amount
        progress_bar(progress)
        similar_frames = []
        similar_frames.append(frame_numbers[frame_index])

        for next_index in range(frame_index + 1, len(frame_numbers)):
            # Load the next frame
            next_frame_path = os.path.join(frames_dir, frame_numbers[next_index] + ".png")
            next_frame = generate_matted_frame(next_frame_path, mask_dir, frame_numbers[next_index])
            
            # Compare the current frame with the next frame
            similarity_score = orb_comparison(base_frame, next_frame)
            if similarity_score > min_similarity_threshold and similarity_score < max_similarity_threshold:
                # If the frames are similar enough, add the next checked frame to the similar_frames list
                similar_frames.append(frame_numbers[next_index])
            else:
                # If the frames are not similar, break out of the inner loop
                break
        
        replace_files_similar_mattes(mask_dir, similar_frames)
        
        # Find the actual index of the last similar frame in the input list and update the frame_index from that point onwards
        last_similar_frame_index = frame_numbers.index(similar_frames[-1])
        frame_index = last_similar_frame_index + 1

        # Check if all the frames have been processed
        if frame_index >= frames_amount:
            progress_bar(1)
            print(f"\nDeduped {deduped_frames_amount} matte frames")
            gr.Info(f"Deduped {deduped_frames_amount} matte frames", duration=5)
            return
        else:
            deduped_frames_amount += (len(similar_frames)-1) # Base frame gets stored in the list as well, hence the subtraction

            # Load the next frame
            new_base_frame_path = os.path.join(frames_dir, frame_numbers[frame_index] + ".png")
            base_frame = generate_matted_frame(new_base_frame_path, mask_dir, frame_numbers[frame_index])