import cv2
from segment_anything import sam_model_registry, SamPredictor
import torch
from matplotlib import pyplot as plt
import numpy as np
import argparse
import os
from utils import show_mask, show_box, check_if_folder_exists
    
    
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", help="Path for image to be segmented")
    parser.add_argument("-f", "--folder", help="Folder where dataset will be created from segmented image")
    args = parser.parse_args()
    
    dest_folder = args.folder
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    # Cargar imagen
    image = cv2.imread(args.image)
    raw_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    predictor.set_image(raw_image)

    # Mostrar imagen
    cv2.imshow("Imagen", image)

    rects = []
    while True:
        # Seleccionar región con un rectángulo
        rect = cv2.selectROI("Imagen", image, False)

        # Mostrar región seleccionada con un rectángulo
        x, y, w, h = rect
        new_rect = [x,y,x+w,y+h]
        rects.append(new_rect)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow("Imagen", image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                # print(rects)
                break

    input_boxes = torch.tensor(rects, device=predictor.device)

    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    # print(masks.shape)

    #Save masks
    check_if_folder_exists(dest_folder)
    i=len(os.listdir(os.path.join(dest_folder,"images")))
    for mask in masks:
        mask_as_numpy = mask.cpu().numpy()[0]
        image_name = f"{i}.png"
        mask_name = f"{i}_seg0.png"
        # print(mask.cpu().numpy())
        cv2.imwrite(os.path.join(dest_folder,"segmentations",mask_name),mask_as_numpy*255)
        cv2.imwrite(os.path.join(dest_folder,"images",image_name), cv2.cvtColor(raw_image,cv2.COLOR_RGB2BGR))
        i+=1
    
    
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box in input_boxes:
        show_box(box.cpu().numpy(), plt.gca())
    plt.axis('off')
    plt.show()
    print(masks)

   