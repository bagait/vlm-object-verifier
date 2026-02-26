import argparse
import os
import json
import requests
from pathlib import Path
from PIL import Image
from groq import Groq
from ultralytics import YOLO
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
IMAGE_URL = "https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg"
ASSETS_DIR = Path("assets")
DEFAULT_IMAGE_PATH = ASSETS_DIR / "dogs.jpg"

# --- LLM for Noun Extraction ---
def extract_objects_with_llm(caption: str) -> list[str]:
    """Uses a fast LLM to extract concrete object nouns from a caption."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")

    client = Groq(api_key=GROQ_API_KEY)
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert noun extractor. Your task is to identify and list all concrete, physical objects mentioned in the user's text. Respond ONLY with a valid JSON list of strings. For example, for the input 'A black cat and a brown dog are sitting on a red couch.', you would output [\"cat\", \"dog\", \"couch\"]."
                },
                {
                    "role": "user",
                    "content": caption,
                },
            ],
            model="llama3-8b-8192",
            temperature=0,
            response_format={"type": "json_object"},
        )
        response_content = chat_completion.choices[0].message.content
        # The model is instructed to return a JSON list, but the API may wrap it in a root key
        # We need to robustly extract the list of strings.
        data = json.loads(response_content)
        
        # Find the list within the parsed JSON
        if isinstance(data, list):
            extracted_list = data
        else:
            # Look for a list in the dictionary values
            extracted_list = []
            for key, value in data.items():
                if isinstance(value, list):
                    extracted_list = value
                    break
            if not extracted_list:
                 raise ValueError("LLM did not return a list of objects in the expected format.")

        return [str(item).lower() for item in extracted_list]

    except Exception as e:
        print(f"Error communicating with Groq API: {e}")
        return []

# --- Vision Model for Object Detection ---
def detect_objects_with_yolo(image_path: Path) -> list[str]:
    """Uses YOLOv8 to detect objects in an image and returns their names."""
    print("\nLoading YOLOv8 model... (This might take a moment on first run)")
    model = YOLO('yolov8n.pt')  # Use the nano model for speed
    
    print(f"Analyzing image with YOLOv8: {image_path}")
    results = model(image_path, verbose=False) # Set verbose to False to reduce console spam
    
    detected_objects = set()
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            detected_objects.add(class_name.lower())
            
    return list(detected_objects)

# --- Verification Logic ---
def verify_caption(image_path: Path, caption: str):
    """Compares objects from a caption against objects detected in an image."""
    print(f"\n--- Verifying Caption --- ")
    print(f'IMAGE: {image_path}')
    print(f'CAPTION: "{caption}"\n')
    
    # Step 1: Extract objects from the caption using LLM
    print("Step 1: Extracting objects from caption using LLM...")
    caption_objects = extract_objects_with_llm(caption)
    if not caption_objects:
        print("Could not extract any objects from the caption.")
        return
    print(f"  > LLM identified: {caption_objects}")

    # Step 2: Detect objects in the image using YOLO
    print("\nStep 2: Detecting objects in the image using YOLOv8...")
    image_objects = detect_objects_with_yolo(image_path)
    if not image_objects:
        print("YOLOv8 did not detect any objects in the image.")
        return
    print(f"  > YOLOv8 detected: {image_objects}")
    
    # Step 3: Compare and report hallucinations
    print("\nStep 3: Comparing results and identifying hallucinations...")
    caption_set = set(caption_objects)
    image_set = set(image_objects)
    
    hallucinated_objects = caption_set - image_set
    verified_objects = caption_set.intersection(image_set)
    
    print("\n--- Verification Report ---")
    if verified_objects:
        print(f"✅ VERIFIED objects: {list(verified_objects)}")
    else:
        print("No objects from the caption were verified in the image.")
        
    if hallucinated_objects:
        print(f"🚨 POTENTIAL HALLUCINATIONS: {list(hallucinated_objects)}")
    else:
        print("🎉 No object hallucinations detected!")
    print("-------------------------")

# --- Setup and Main Execution ---
def setup_environment():
    """Creates assets directory and downloads the sample image if needed."""
    ASSETS_DIR.mkdir(exist_ok=True)
    if not DEFAULT_IMAGE_PATH.exists():
        print(f"Downloading sample image to {DEFAULT_IMAGE_PATH}...")
        try:
            response = requests.get(IMAGE_URL, stream=True)
            response.raise_for_status()
            with open(DEFAULT_IMAGE_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading image: {e}")
            exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Verify objects in an image caption against a YOLOv8 object detection model.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--image", 
        type=str, 
        default=str(DEFAULT_IMAGE_PATH),
        help="Path to the image file."
    )
    parser.add_argument(
        "--caption", 
        type=str, 
        required=True, 
        help="The caption to verify."
    )
    
    args = parser.parse_args()
    
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image file not found at {image_path}")
        return

    verify_caption(image_path, args.caption)

if __name__ == "__main__":
    setup_environment()
    main()