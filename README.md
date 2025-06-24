# 🔍 Smart Image Search & Tagging Web App

Ein GPU-beschleunigtes Tool zur visuellen Bildersuche, automatischen Caption-Erstellung, Tagging und Website-Bild-Scraping – alles in einer Weboberfläche.

![Screenshot 2025-06-24 230457](https://github.com/user-attachments/assets/98259556-ac3b-4b34-8555-63892fa28525)

## ✨ Features

- Drag & Drop oder Upload für Bildsuche
- Textsuche mit natürlichen Beschreibungen (z. B. „a dog in the snow“)
- Automatische Captions mit [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base)
- Tag-Extraktion mit [spaCy](https://spacy.io/)
- Ähnlichkeitssuche über [CLIP](https://github.com/openai/CLIP)
- GPU-Unterstützung (CUDA, wenn verfügbar)
- Website-Scraper: lädt und verarbeitet automatisch alle Bilder einer URL
- Schlagwortsuche per Klick auf Tags
- Responsive Web UI mit Lightbox, Vorschau & Galerie

![Screenshot 2025-06-24 230302](https://github.com/user-attachments/assets/6a156140-503e-4441-9250-87641023a379)

## 🚀 Schnellstart

```bash
git clone https://github.com/JonPlaytv/imagedb.git
cd image-search-app
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python app.py
