import os
import json
import numpy as np
from PIL import Image
import imagehash
import clip
import torch
import spacy
from transformers import BlipProcessor, BlipForConditionalGeneration

# Paths
IMAGE_FOLDER = r"E:\imagedb\static\images"
EMBEDDINGS_FILE = "clip_embeddings.npy"
METADATA_FILE = "metadata.json"

os.makedirs(IMAGE_FOLDER, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
nlp = spacy.load("en_core_web_sm")

def generate_caption(image: Image.Image) -> str:
    inputs = processor(image, return_tensors="pt").to(device)
    outputs = caption_model.generate(**inputs)
    return processor.decode(outputs[0], skip_special_tokens=True)

def extract_keywords(caption: str):
    doc = nlp(caption.lower())
    return list(set(token.lemma_ for token in doc if token.is_alpha and not token.is_stop))

def process_image(img: Image.Image, filename: str):
    img = img.convert("RGB")
    img = img.resize((512, 512))
    path = os.path.join(IMAGE_FOLDER, filename)
    img.save(path, "WEBP", quality=85)  # overwrite with webp version if needed

    image_input = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image_input).cpu().numpy()[0]

    caption = generate_caption(img)
    tags = extract_keywords(caption)
    phash = str(imagehash.phash(img))

    return embedding, {
        "path": os.path.join("static/images", filename).replace("\\", "/"),
        "caption": caption,
        "tags": tags,
        "hash": phash,
        "source_url": None
    }

def load_existing_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return []

def load_existing_embeddings():
    if os.path.exists(EMBEDDINGS_FILE):
        return np.load(EMBEDDINGS_FILE)
    return np.empty((0, 512))  # Assuming CLIP ViT-B/32 embeddings are 512-dim

def main():
    metadata = load_existing_metadata()
    embeddings = load_existing_embeddings()

    # To speed up lookup, store existing filenames and hashes
    existing_files = set(item["path"].split("/")[-1] for item in metadata)
    existing_hashes = set(item.get("hash") for item in metadata if "hash" in item)

    new_embeddings = []
    new_metadata = []

    for filename in os.listdir(IMAGE_FOLDER):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")):
            continue

        if filename in existing_files:
            print(f"Skipping existing image by filename: {filename}")
            continue

        filepath = os.path.join(IMAGE_FOLDER, filename)
        try:
            img = Image.open(filepath)
        except Exception as e:
            print(f"Cannot open {filename}: {e}")
            continue

        # Compute perceptual hash and skip if hash exists (extra safety)
        img_hash = str(imagehash.phash(img))
        if img_hash in existing_hashes:
            print(f"Skipping existing image by hash: {filename}")
            continue

        print(f"Processing {filename} ...")
        embedding, meta = process_image(img, filename)

        new_embeddings.append(embedding)
        new_metadata.append(meta)
        existing_files.add(filename)
        existing_hashes.add(img_hash)

    if new_embeddings:
        all_embeddings = np.vstack([embeddings, new_embeddings])
        np.save(EMBEDDINGS_FILE, all_embeddings)

        all_metadata = metadata + new_metadata
        with open(METADATA_FILE, "w") as f:
            json.dump(all_metadata, f, indent=2)

        print(f"✅ Processed {len(new_embeddings)} new images.")
    else:
        print("No new images to process.")

if __name__ == "__main__":
    main()
