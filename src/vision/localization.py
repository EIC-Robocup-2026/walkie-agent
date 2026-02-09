import torch
import cv2
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from transformers import BlipProcessor, BlipForConditionalGeneration
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
import requests

# -------------------------
# Helper to download assets if missing
# -------------------------
def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filename}.")
    else:
        print(f"{filename} already exists.")

# Download SAM checkpoint
sam_checkpoint = "sam_vit_b_01ec64.pth"
sam_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
download_file(sam_url, sam_checkpoint)

# Download sample image if room.jpg doesn't exist
image_path = "room.jpg"
if not os.path.exists(image_path):
    # downloading a sample living room image
    image_url = "https://images.pexels.com/photos/1571460/pexels-photo-1571460.jpeg" 
    download_file(image_url, image_path)

# -------------------------
# Load Models
# -------------------------

print("Loading models...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# CLIP
print("Loading CLIP...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# BLIP (Base)
print("Loading BLIP...")
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    torch_dtype=torch.float16 if device=="cuda" else torch.float32
)
caption_model.to(device)

# SAM
print("Loading SAM...")
sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
sam.to(device)
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.90,       # Require high overlap confidence
    stability_score_thresh=0.95, # Require stable masks
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=200,   # Ignore tiny specks
)

# -------------------------
# Load Image
# -------------------------

print(f"Loading image from {image_path}...")
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

pil_image = Image.fromarray(image_rgb)

# -------------------------
# Generate Masks (Objects)
# -------------------------

print("Generating masks...")
masks = mask_generator.generate(image_rgb)

print(f"Detected {len(masks)} objects")

# -------------------------
# Dynamic Room Classification
# -------------------------
print("Classifying room type...")
possible_rooms = ["kitchen", "living room", "bedroom", "bathroom", "office", "dining room", "corridor"]
# CLIP expects text and images together for zero-shot classification
inputs = clip_processor(text=possible_rooms, images=pil_image, return_tensors="pt", padding=True).to(device)

with torch.no_grad():
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image # image-text similarity
    probs = logits_per_image.softmax(dim=1)

best_match_idx = probs.argmax().item()
room_label = possible_rooms[best_match_idx]
print(f"Detected Room: {room_label} (Confidence: {probs[0, best_match_idx].item():.4f})")

results = []

# -------------------------
# Process each object
# -------------------------

print("Processing objects...")
total_area = image_rgb.shape[0] * image_rgb.shape[1]

# Sort masks by size (largest first) to prioritize main objects
masks = sorted(masks, key=lambda x: x['area'], reverse=True)
MAX_OBJECTS = 3

for i, mask in enumerate(masks):
    if len(results) >= MAX_OBJECTS:
        break

    # Filter by area: Ignore < 0.5% (noise) and > 50% (likely walls/floors)
    area_ratio = mask['area'] / total_area
    if area_ratio < 0.005 or area_ratio > 0.50:
        continue

    seg = mask["segmentation"]

    # crop object
    ys, xs = np.where(seg)
    if len(xs) == 0:
        continue

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    # Expand crop by 50px for context
    pad = 50
    h, w, _ = image_rgb.shape
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    # Ensure valid slice
    if x2 <= x1 or y2 <= y1:
        continue

    crop = image_rgb[y1:y2, x1:x2]
    crop_pil = Image.fromarray(crop)

    # ---- CLIP embedding ----
    clip_inputs = clip_processor(images=crop_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        clip_features = clip_model.get_image_features(**clip_inputs)
        clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)

    # ---- Caption ----
    caption_inputs = caption_processor(images=crop_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        caption_ids = caption_model.generate(**caption_inputs, max_new_tokens=20)

    caption = caption_processor.decode(caption_ids[0], skip_special_tokens=True)

    # store result
    results.append({
        "object_id": i,
        "room": room_label,
        "caption": caption,
        "embedding": clip_features.cpu().numpy()
    })

    print(f"[{i}] Room: {room_label} | Caption: {caption}")

# -------------------------
# Example query
# -------------------------

query = "chair in the kitchen"
print(f"Querying for: '{query}'")
text_inputs = clip_processor(text=query, return_tensors="pt").to(device)

with torch.no_grad():
    text_feat = clip_model.get_text_features(**text_inputs)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

# cosine similarity search
print("\nSearch Results:")
scores = []
for r in results:
    sim = np.dot(text_feat.cpu().numpy(), r["embedding"].T).item()
    scores.append((sim, r))
    # print(f"Match score with object {r['object_id']} ({r['caption']}): {sim}")

# Sort by score
scores.sort(key=lambda x: x[0], reverse=True)
for sim, r in scores[:5]: # Show top 5
    print(f"Score: {sim:.4f} | Object {r['object_id']} | Caption: {r['caption']}")