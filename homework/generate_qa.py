import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# -------------------------------
# NORMALIZATION FUNCTION (NEW)
# -------------------------------
def normalize_direction(answer):
    """
    Normalize direction wording to EXACT grader format:
    - front/back comes first
    - left/right comes second
    - converts "in front of" → "front"
    - converts "behind" → "back"
    - converts "right and back" → "back and right"
    """
    answer = answer.strip().lower()

    # canonical mapping
    canonical = {
        "front": "front",
        "in front": "front",
        "in front of": "front",

        "behind": "back",
        "in back": "back",
        "back": "back",

        "left": "left",
        "right": "right",
    }

    # If exact match
    if answer in canonical:
        return canonical[answer]

    # Split combined directions
    parts = [p.strip() for p in answer.replace("of", "").split("and")]
    mapped = [canonical.get(p, p) for p in parts]

    fb = None
    lr = None
    for p in mapped:
        if p in ("front", "back"):
            fb = p
        elif p in ("left", "right"):
            lr = p

    if fb and lr:
        return f"{fb} and {lr}"

    return answer


# -------------------------------
# CONSTANTS
# -------------------------------
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

COLORS = {
    1: (0, 255, 0),
    2: (255, 0, 0),
    3: (0, 0, 255),
    4: (255, 255, 0),
    5: (255, 0, 255),
    6: (0, 255, 255),
}

ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


# -------------------------------
# FRAME INFO PARSING
# -------------------------------
def extract_frame_info(image_path: str) -> tuple[int, int]:
    filename = Path(image_path).name
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0


# -------------------------------
# DRAW DETECTIONS FOR DEBUGGING
# -------------------------------
def draw_detections(image_path: str, info_path: str, font_scale=0.5, thickness=1, min_box_size=5) -> np.ndarray:

    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    img_width, img_height = pil_image.size
    draw = ImageDraw.Draw(pil_image)

    with open(info_path) as f:
        info = json.load(f)

    _, view_index = extract_frame_info(image_path)

    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        return np.array(pil_image)

    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        if (x2_scaled - x1_scaled < min_box_size) or (y2_scaled - y1_scaled < min_box_size):
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        color = COLORS.get(class_id, (255, 255, 255))
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    return np.array(pil_image)


# -------------------------------
# KART EXTRACTION
# -------------------------------
def extract_kart_objects(info_path: str, view_index: int, img_width=150, img_height=100, min_box_size=5) -> list:

    with open(info_path) as f:
        info = json.load(f)

    if view_index >= len(info["detections"]):
        return []

    frame_detections = info["detections"][view_index]

    kart_objects = []
    image_center = (img_width / 2, img_height / 2)
    min_distance_to_center = float("inf")
    center_kart_id = None

    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        if (x2_scaled - x1_scaled < min_box_size) or (y2_scaled - y1_scaled < min_box_size):
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # NOTE: keeping original behavior as requested (still uses scaled center)
        center_x = (x1_scaled + x2_scaled) / 2
        center_y = (y1_scaled + y2_scaled) / 2

        dist = np.sqrt((center_x - image_center[0])**2 + (center_y - image_center[1])**2)
        if dist < min_distance_to_center:
            min_distance_to_center = dist
            center_kart_id = track_id

        karts = info["karts"]
        kart_objects.append({
            "instance_id": track_id,
            "kart_name": karts[track_id],
            "center": (center_x, center_y)
        })

    for kart in kart_objects:
        kart["is_center_kart"] = (kart["instance_id"] == center_kart_id)

    return kart_objects


# -------------------------------
# TRACK EXTRACTION
# -------------------------------
def extract_track_info(info_path: str) -> str:
    with open(info_path) as f:
        info = json.load(f)
    return info.get("track", "Unknown Track")


# -------------------------------
# QA PAIR GENERATION
# -------------------------------
def generate_qa_pairs(info_path: str, view_index: int, img_width=150, img_height=100):

    qa_pairs = []

    kart_objects = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)

    # Ego kart
    center_kart = next((k for k in kart_objects if k["is_center_kart"]), None)
    if center_kart:
        qa_pairs.append({
            "question": "What kart is the ego car?",
            "answer": center_kart["kart_name"]
        })

    # Count
    qa_pairs.append({
        "question": "How many karts are there in the scenario?",
        "answer": str(len(kart_objects))
    })

    # Track
    qa_pairs.append({
        "question": "What track is this?",
        "answer": track_name
    })

    if center_kart:
        center_x, center_y = center_kart["center"]

        for kart in kart_objects:
            if kart["is_center_kart"]:
                continue

            kart_x, kart_y = kart["center"]

            lr = "left" if kart_x < center_x else "right"
            fb = "front" if kart_y < center_y else "back"

            # Normalize all direction answers
            lr = normalize_direction(lr)
            fb = normalize_direction(fb)
            combined = normalize_direction(f"{fb} and {lr}")

            # Left/Right question
            qa_pairs.append({
                "question": f"Is {kart['kart_name']} to the left or right of the ego car?",
                "answer": lr
            })

            # Front/Behind
            qa_pairs.append({
                "question": f"Is {kart['kart_name']} in front of or behind the ego car?",
                "answer": fb
            })

            # Relative position
            qa_pairs.append({
                "question": f"Where is {kart['kart_name']} relative to the ego car?",
                "answer": combined
            })

        # Counting
        left_count = sum(1 for k in kart_objects if not k["is_center_kart"] and k["center"][0] < center_x)
        right_count = sum(1 for k in kart_objects if not k["is_center_kart"] and k["center"][0] >= center_x)
        front_count = sum(1 for k in kart_objects if not k["is_center_kart"] and k["center"][1] < center_y)
        back_count = sum(1 for k in kart_objects if not k["is_center_kart"] and k["center"][1] >= center_y)

        qa_pairs.append({"question": "How many karts are to the left of the ego car?", "answer": str(left_count)})
        qa_pairs.append({"question": "How many karts are to the right of the ego car?", "answer": str(right_count)})
        qa_pairs.append({"question": "How many karts are in front of the ego car?", "answer": str(front_count)})
        qa_pairs.append({"question": "How many karts are behind the ego car?", "answer": str(back_count)})

    return qa_pairs


# -------------------------------
# DEBUG VISUALIZER
# -------------------------------
def check_qa_pairs(info_file: str, view_index: int):

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    qa_pairs = generate_qa_pairs(info_file, view_index)

    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


def main():
    fire.Fire({"check": check_qa_pairs})


if __name__ == "__main__":
    main()
