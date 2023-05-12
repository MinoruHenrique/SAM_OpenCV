import numpy as np
from matplotlib import pyplot as plt
import os
import cv2
import torch
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    
def check_if_folder_exists(folder):
    #First check if root folder doesn't exists
    if(not os.path.exists(folder)):
        print(f"The path {folder} doesn't exist. Creating {folder}...")
        os.mkdir(folder)
    #Then, there have to exist folders "images" and "segmentations"
    images_folder = os.path.join(folder, "images")
    segm_folder = os.path.join(folder,"segmentations")
    if(not os.path.exists(images_folder)):
        print(f"The path {images_folder} doesn't exist. Creating {images_folder}...")
        os.mkdir(images_folder)
    if(not os.path.exists(segm_folder)):
        print(f"The path {segm_folder} doesn't exist. Creating {segm_folder}...")
        os.mkdir(segm_folder)
def do_segmentation(predictor, image):
    predictor.set_image(image)
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
    return masks, input_boxes
def save_masks(dest_folder, image, masks):
    i=len(os.listdir(os.path.join(dest_folder,"images")))
    for mask in masks:
        mask_as_numpy = mask.cpu().numpy()[0]
        image_name = f"{i}.png"
        mask_name = f"{i}_seg0.png"
        # print(mask.cpu().numpy())
        cv2.imwrite(os.path.join(dest_folder,"segmentations",mask_name),mask_as_numpy*255)
        cv2.imwrite(os.path.join(dest_folder,"images",image_name), image)
        i+=1
def show_segmentation(image, masks, input_boxes):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box in input_boxes:
        show_box(box.cpu().numpy(), plt.gca())
    plt.axis('off')
    plt.show()