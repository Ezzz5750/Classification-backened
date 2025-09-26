# utils/segmenter.py
import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from groundingdino.util.inference import load_model, predict
import os

# Paths
SAM_CHECKPOINT = "models/sam_vit_b_01ec64.pth"
GROUNDING_DINO_CONFIG = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "models/groundingdino_swint_ogc.pth"

# Lazy load models
_dino_model = None
_sam_predictor = None

def get_dino_model():
    global _dino_model
    if _dino_model is None:
        _dino_model = load_model(GROUNDING_DINO_CONFIG, GROUNDING_DINO_CHECKPOINT)
    return _dino_model

def get_sam_predictor():
    global _sam_predictor
    if _sam_predictor is None:
        sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
        sam.eval()  # No CUDA
        _sam_predictor = SamPredictor(sam)
    return _sam_predictor

def segment_food(image, text_prompt, box_threshold=0.3, text_threshold=0.25):
    """
    Returns binary mask (H, W) where food is detected.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    dino_model = get_dino_model()
    boxes, logits, phrases = predict(
        model=dino_model,
        image=image_pil,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    if len(boxes) == 0:
        return None

    predictor = get_sam_predictor()
    predictor.set_image(image_rgb)

    # Use first box
    input_box = boxes[0].cpu().numpy() * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
    masks, _, _ = predictor.predict(box=input_box)

    return masks[0]  # H x W bool array