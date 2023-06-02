import cv2
from segment_anything import sam_model_registry, SamPredictor
import argparse
import os
from utils import check_if_folder_exists, do_segmentation, save_masks, show_segmentation
    
_AVAILABLE_MODES = ["bb","pt"]

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--data", help="Path for image or image folder to be segmented")
    parser.add_argument("-f", "--folder", help="Folder where dataset will be created from segmented image")
    parser.add_argument("-m", "--mode", help="Use mode (bb for bounding boxes/ pt for point selection)")
    args = parser.parse_args()
    
    mode = args.mode
    
    if mode not in _AVAILABLE_MODES:
        raise ValueError(f"Mode '{mode}' not supported")
    
    dest_folder = args.folder
    check_if_folder_exists(dest_folder)
    
    print("Loading SAM weights...")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
   
    # Single photo
    if(os.path.isfile(args.data)):
        image = cv2.imread(args.data)
        raw_image = image.copy()
        masks, input_boxes = do_segmentation(predictor, image, mode)
        #Save masks
        save_masks(dest_folder,raw_image,masks,mode)
        #Show segmentation as a matplotlib plot
        show_segmentation(raw_image, masks, input_boxes, mode)
    # Folder of photos
    elif(os.path.isdir(args.data)):
        images = os.listdir(args.data)
        for image in images:
            image_path = os.path.join(args.data, image)
            image = cv2.imread(image_path)
            raw_image = image.copy()
            masks, input_boxes = do_segmentation(predictor, image, mode)
            #Save masks
            save_masks(dest_folder, raw_image, masks,mode)
            
            #Show segmentation as a matplotlib plot
            show_segmentation(raw_image, masks, input_boxes, mode)
            

   