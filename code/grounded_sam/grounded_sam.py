import torch
import requests
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from segment_anything import sam_model_registry, SamPredictor

# -------- Load Grounding DINO --------
model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

repo_root = Path.cwd()
IMAGE_NAME = repo_root / "data" / "test_images" /"resized_test_fire_frame43.jpg"
image = Image.open(IMAGE_NAME)

# Detection text prompt
text_prompt = "flame. fire. smoke."

inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

# Post-process boxes
results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    threshold=0.4,
    text_threshold=0.25,
    target_sizes=[image.size[::-1]],
)

boxes = [r["boxes"].cpu().numpy() for r in results]
labels = [r["labels"] for r in results]  # all should be "fire"
print(f"Found {len(boxes[0])} fire objects")

# -------- Load SAM --------
sam_checkpoint = repo_root / "externals" / "sam_vit_h_4b8939.pth"
sam_model = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
sam_model.to(device)
predictor = SamPredictor(sam_model)
predictor.set_image(np.array(image))

# -------- Get masks --------
for i, box in enumerate(boxes[0]):
    masks, scores, logits = predictor.predict(box=box, multimask_output=False)
    mask = masks[0]  # single mask

    # -------- Visualization --------
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.imshow(mask, alpha=0.5, cmap="Reds")
    plt.title(f"Fire instance {i+1}")
    plt.axis("off")
    plt.show()
