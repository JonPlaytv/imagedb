import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from PIL import Image
from io import BytesIO
import imagehash
import clip
import torch
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
import spacy
import json

SAVE_FOLDER = "static/images"
EMBEDDINGS_FILE = "clip_embeddings.npy"
METADATA_FILE = "metadata.json"

os.makedirs(SAVE_FOLDER, exist_ok=True)

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

def get_image_urls_from_website(base_url):
    try:
        resp = requests.get(base_url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        img_urls = set()
        for img in soup.find_all("img"):
            src = img.get("src")
            if src:
                full_url = urljoin(base_url, src)
                if full_url.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")):
                    img_urls.add(full_url)
        return list(img_urls)
    except Exception as e:
        print(f"[ERROR] Failed to scrape {base_url}: {e}")
        return []

def download_image(url, idx):
    try:
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img = img.resize((512, 512))
        filename = f"site_{idx:03d}.webp"
        path = os.path.join(SAVE_FOLDER, filename)
        img.save(path, "WEBP", quality=85)
        return img, filename
    except Exception as e:
        print(f"[SKIP] {url}: {e}")
        return None, None

def scrape_site_and_process(base_url):
    urls = get_image_urls_from_website(base_url)
    embeddings = []
    metadata = []

    for idx, url in enumerate(urls):
        img, filename = download_image(url, idx)
        if img is None:
            continue

        image_input = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image_input).cpu().numpy()[0]
        caption = generate_caption(img)
        tags = extract_keywords(caption)
        phash = str(imagehash.phash(img))

        embeddings.append(embedding)
        metadata.append({
            "path": f"{SAVE_FOLDER}/{filename}",
            "caption": caption,
            "tags": tags,
            "hash": phash,
            "source_url": url
        })

    # Save embeddings
    if os.path.exists(EMBEDDINGS_FILE):
        existing = np.load(EMBEDDINGS_FILE)
        embeddings = np.vstack([existing, embeddings])
    else:
        embeddings = np.array(embeddings)
    np.save(EMBEDDINGS_FILE, embeddings)

    # Save metadata
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            existing = json.load(f)
        metadata = existing + metadata
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Scraped and processed {len(metadata)} total entries from {base_url}.")

# Example usage:
scrape_site_and_process("https://bodensee-gymnasium.de")
