import os
import re
import json
import numpy as np
import clip
import torch
import imagehash
import requests
import mimetypes
import cv2
import tempfile
import pillow_avif
import tempfile
from flask import redirect, url_for
from flask import jsonify

from PIL import Image, ImageSequence
from io import BytesIO
from bs4 import BeautifulSoup
from flask import Flask, request, render_template
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BlipProcessor, BlipForConditionalGeneration
import spacy
from urllib.parse import urljoin

app = Flask(__name__)
IMAGE_FOLDER = "static/images"
EMBEDDINGS_FILE = "clip_embeddings.npy"
METADATA_FILE = "metadata.json"

os.makedirs(IMAGE_FOLDER, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
nlp = spacy.load("en_core_web_sm")

image_ext_pattern = re.compile(r"\.(jpg|jpeg|png|webp|bmp|gif|avif|mp4|webm)$", re.IGNORECASE)

if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(METADATA_FILE):
    clip_vectors = np.load(EMBEDDINGS_FILE)
    with open(METADATA_FILE) as f:
        metadata = json.load(f)
else:
    clip_vectors = np.zeros((0, 512), dtype=np.float32)
    metadata = []

print(f"[INIT] Loaded {len(metadata)} metadata items with {clip_vectors.shape[0]} vectors.")

# --- IMAGE LOADING / PREPROCESSING ---
def load_image_safely(file_stream, filename=None):
    try:
        ext = os.path.splitext(filename)[-1].lower() if filename else ""

        if ext in [".mp4", ".webm"]:
            # Video: temporär speichern und ersten Frame lesen
            with tempfile.NamedTemporaryFile(delete=True, suffix=ext) as tmp:
                tmp.write(file_stream.read())
                tmp.flush()
                video = cv2.VideoCapture(tmp.name)
                ret, frame = video.read()
                video.release()
                if not ret:
                    raise ValueError("No frame extracted from video")
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            # Bild oder GIF
            image = Image.open(file_stream)
            if getattr(image, "is_animated", False):
                image = next(ImageSequence.Iterator(image))  # erster Frame
            image = image.convert("RGB")

        # Aspect Ratio beibehalten und max 512x512
        max_size = 512
        ratio = min(max_size / image.width, max_size / image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        return image.resize(new_size, Image.LANCZOS)

    except Exception as e:
        print(f"[ERROR] load_image_safely: {e}")
        return None
def load_video_frame(file_stream):
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
            tmp.write(file_stream.read())
            tmp.flush()
            video = cv2.VideoCapture(tmp.name)
            ret, frame = video.read()
            video.release()
            if not ret:
                raise ValueError("No frame extracted from video")
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return image
    except Exception as e:
        print(f"[ERROR] load_video_frame: {e}")
        return None




# --- SEARCH / CAPTION ---
def generate_caption(image: Image.Image) -> str:
    inputs = processor(image, return_tensors="pt").to(device)
    outputs = caption_model.generate(**inputs)
    return processor.decode(outputs[0], skip_special_tokens=True)

def extract_keywords(caption: str):
    doc = nlp(caption.lower())
    return list(set(token.lemma_ for token in doc if token.is_alpha and not token.is_stop))

def jaccard_similarity(tags1, tags2):
    set1, set2 = set(tags1), set(tags2)
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

def process_query_image(image):
    query_hash = imagehash.phash(image)
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        query_embedding = model.encode_image(image_input).cpu().numpy()
    return query_hash, query_embedding

def process_query_text(text):
    text_input = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_embedding = model.encode_text(text_input).cpu().numpy()
    return text_embedding

# --- CRAWLER ---
def crawl_images(base_url, depth=4, max_pages=20, visited=None):
    if visited is None:
        visited = set()
    if base_url in visited or len(visited) >= max_pages:
        return []
    visited.add(base_url)

    try:
        resp = requests.get(base_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        img_urls = set()
        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src")
            if src:
                full_url = urljoin(base_url, src)
                if image_ext_pattern.search(full_url):
                    img_urls.add(full_url)

        for link in soup.select("a[href]"):
            href = urljoin(base_url, link['href'])
            if href not in visited and base_url.split("//")[1].split("/")[0] in href:
                img_urls.update(crawl_images(href, depth-1, max_pages, visited))
                if len(visited) >= max_pages:
                    break

        return list(img_urls)
    except Exception as e:
        print(f"[ERROR] crawl {base_url}: {e}")
        return []

# --- DOWNLOAD ---
def download_and_embed(input_data):
    source = "<unknown>"  # Default-Wert zu Beginn definieren
    try:
        if isinstance(input_data, str):
            source = input_data
            r = requests.get(input_data, timeout=10)
            r.raise_for_status()
            image = load_image_safely(BytesIO(r.content), input_data)
        else:
            source = getattr(input_data, 'filename', None) or "uploaded_file"
            image = load_image_safely(input_data.stream if hasattr(input_data, 'stream') else input_data, source)

        if image is None:
            return None, None

        phash = str(imagehash.phash(image))
        if any(m.get("hash") == phash for m in metadata):
            print(f"[SKIP] Duplicate image (hash: {phash}) from {source}")
            return None, None

        filename = f"{phash}.webp"
        path = os.path.join(IMAGE_FOLDER, filename)
        image.save(path, "WEBP", quality=85)

        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image_input).cpu().numpy()[0]

        caption = generate_caption(image)
        tags = extract_keywords(caption)

        meta = {
            "path": f"{IMAGE_FOLDER}/{filename}".replace("\\", "/"),
            "caption": caption,
            "tags": tags,
            "hash": phash,
            "source": source
        }

        metadata.append(meta)
        global clip_vectors
        clip_vectors = np.vstack([clip_vectors, embedding[None]])
        return meta, embedding

    except Exception as e:
        print(f"[SKIP] {source}: {e}")
        return None, None






@app.route("/", methods=["GET", "POST"])
def index():
    matches = []
    generated_caption = None
    generated_tags = []
    query_type = None

    if request.method == "POST":
        file = request.files.get("query")
        text = request.form.get("text_query", "").strip()
        site_url = request.form.get("site_query", "").strip()

        if file and file.filename:
            query_type = "image"
            query_img = load_image_safely(file.stream, file.filename)
            if query_img:
                generated_caption = generate_caption(query_img)
                generated_tags = extract_keywords(generated_caption)
                query_hash, query_embed = process_query_image(query_img)

                # Filter indices mit Sicherheitsschicht gegen out-of-range Indizes
                filtered_indices = [
                    i for i, meta in enumerate(metadata)
                    if "hash" in meta
                    and (query_hash - imagehash.hex_to_hash(meta["hash"])) < 16
                    and i < clip_vectors.shape[0]
                ]

                # Falls zu wenig Treffer, fallback auf alle Indizes im gültigen Bereich
                if not filtered_indices:
                    filtered_indices = list(range(min(len(metadata), clip_vectors.shape[0])))

                search_vectors = clip_vectors[filtered_indices]
                sims = cosine_similarity(query_embed, search_vectors)[0]

                results = []
                for i, sim in enumerate(sims):
                    idx = filtered_indices[i]
                    meta = metadata[idx]
                    tag_score = jaccard_similarity(generated_tags, meta.get("tags", []))
                    combined = 0.6 * sim + 0.4 * tag_score
                    results.append((combined, meta))

                results.sort(key=lambda x: x[0], reverse=True)
                for score, meta in results[:50]:
                    matches.append((
                        meta["path"], 0, score,
                        meta.get("caption", ""),
                        meta.get("tags", []),
                        meta.get("source", "")
                    ))

        elif text:
            query_type = "text"
            query_embed = process_query_text(text)
            sims = cosine_similarity(query_embed, clip_vectors)[0]
            top_indices = np.argsort(sims)[-100:][::-1]

            for idx in top_indices:
                meta = metadata[idx]
                if text.lower() in meta.get("caption", "").lower() or any(text.lower() in tag for tag in meta.get("tags", [])):
                    matches.append((
                        meta["path"], 0, sims[idx],
                        meta.get("caption", ""),
                        meta.get("tags", []),
                        meta.get("source", "")
                    ))
                    if len(matches) >= 50:
                        break

        elif site_url:
            query_type = "url"
            img_urls = crawl_images(site_url)
            for url in img_urls[:15]:
                meta, _ = download_and_embed(url)
                if meta:
                    matches.append((
                        meta["path"], 0, 1.0,
                        meta["caption"],
                        meta["tags"],
                        meta.get("source", "")
                    ))
            # Speichern nach Crawl
            np.save(EMBEDDINGS_FILE, clip_vectors)
            with open(METADATA_FILE, "w") as f:
                json.dump(metadata, f, indent=2)

    return render_template(
        "index.html",
        matches=matches,
        generated_caption=generated_caption,
        generated_tags=generated_tags,
        query_type=query_type,
    )

@app.route("/upload", methods=["POST"])
def upload_files():
    if "files[]" not in request.files:
        return jsonify({"error": "No files part"}), 400

    files = request.files.getlist("files[]")
    results = []

    for file in files:
        if file and file.filename:
            meta, embedding = download_and_embed(file)
            if meta:
                results.append({
                    "path": meta["path"],
                    "caption": meta["caption"],
                    "tags": meta["tags"],
                    "source": meta.get("source", ""),
                })

    # Nach allen Uploads Daten speichern
    np.save(EMBEDDINGS_FILE, clip_vectors)
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    if results:
        return jsonify({"message": f"{len(results)} files uploaded", "results": results})
    else:
        return jsonify({"message": "No valid files uploaded"}), 400






if __name__ == "__main__":
    app.run(debug=True)
