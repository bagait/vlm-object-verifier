# VLM Object Verifier

A tool to detect object hallucinations in Vision-Language Model (VLM) outputs using an independent object detection model as a source of truth.

This project addresses a critical problem in multimodal AI: models often generate descriptions that include objects not present in the image. This tool provides a simple way to verify the factual grounding of an image caption by comparing its content against the robust detections of a YOLOv8 model.



## Features

-   **Hallucination Detection**: Pinpoints specific objects mentioned in a caption that are likely not in the image.
-   **Two-Model Approach**: Uses a fast LLM (via Groq) for intelligent noun extraction from text and a separate computer vision model (YOLOv8) for objective analysis of the image.
-   **Reliable Ground Truth**: Leverages the pre-trained, powerful YOLOv8 object detection model to create a reliable list of objects actually present.
-   **Simple CLI**: Easy-to-use command-line interface for quick verification.

## How It Works

The verification process involves three main steps:

1.  **Noun Extraction (LLM)**: The input caption is sent to a fast Large Language Model (Llama 3 via the Groq API). The LLM is prompted to act as an expert noun extractor, identifying and returning a JSON list of all concrete, physical objects mentioned in the text.

2.  **Object Detection (YOLOv8)**: The input image is analyzed by YOLOv8, a state-of-the-art object detection model. YOLO scans the image and produces a list of all objects it confidently identifies.

3.  **Comparison and Reporting**: The list of objects from the caption (Step 1) is compared against the list of objects detected in the image (Step 2). Any object mentioned in the caption but not found by YOLO is flagged as a potential hallucination.

## Installation

1.  **Clone the repository:**
    bash
    git clone https://github.com/bagait/vlm-object-verifier.git
    cd vlm-object-verifier
    

2.  **Install Python dependencies:**
    bash
    pip install -r requirements.txt
    

3.  **Set up your API Key:**
    -   Get a free API key from [Groq](https://console.groq.com/keys).
    -   Create a file named `.env` in the project root.
    -   Add your API key to the `.env` file:
        
        GROQ_API_KEY="gsk_YourSecretKeyGoesHere"
        

## Usage

The script will automatically download a sample image (`assets/dogs.jpg`) on its first run.

### Example 1: No Hallucination

Let's test a caption that accurately describes the image.

bash
python main.py --caption "A photo of two beautiful dogs sitting in the grass."


**Expected Output:**

--- Verifying Caption --- 
IMAGE: assets/dogs.jpg
CAPTION: "A photo of two beautiful dogs sitting in the grass."

Step 1: Extracting objects from caption using LLM...
  > LLM identified: ['dog']

Step 2: Detecting objects in the image using YOLOv8...
Loading YOLOv8 model... (This might take a moment on first run)
Analyzing image with YOLOv8: assets/dogs.jpg
  > YOLOv8 detected: ['dog']

Step 3: Comparing results and identifying hallucinations...

--- Verification Report ---
✅ VERIFIED objects: ['dog']
🎉 No object hallucinations detected!
-------------------------


### Example 2: Detecting a Hallucination

Now, let's use a caption that includes an object not present in the image: a "cat".

bash
python main.py --caption "A picture of two dogs and a cat relaxing outside."


**Expected Output:**

--- Verifying Caption --- 
IMAGE: assets/dogs.jpg
CAPTION: "A picture of two dogs and a cat relaxing outside."

Step 1: Extracting objects from caption using LLM...
  > LLM identified: ['dog', 'cat']

Step 2: Detecting objects in the image using YOLOv8...
Loading YOLOv8 model... 
Analyzing image with YOLOv8: assets/dogs.jpg
  > YOLOv8 detected: ['dog']

Step 3: Comparing results and identifying hallucinations...

--- Verification Report ---
✅ VERIFIED objects: ['dog']
🚨 POTENTIAL HALLUCINATIONS: ['cat']
-------------------------


The tool correctly identifies that "dog" is present but flags "cat" as a potential hallucination.

## License

This project is licensed under the MIT License. See the LICENSE file for details.