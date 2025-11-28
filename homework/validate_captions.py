import json
from pathlib import Path


def validate_json():
    """
    Compare generated captions in data/valid against the balanced captions
    in data/valid_grader/all_mc_qas.json.
    """
    data_root = Path(__file__).parent.parent
    golden_file = data_root / "data" / "valid_grader" / "all_mc_qas.json"

    with golden_file.open() as f:
        captions_balanced = json.load(f)

    captions_generated: list[dict[str, str]] = []
    captions_files = list((data_root / "data" / "valid").glob("*_captions.json"))

    for caption_file in captions_files:
        with caption_file.open() as f:
            captions_generated.extend(json.load(f))

    print(f"Number of captions golden: {len(captions_balanced)}")
    print(f"Number of captions generated: {len(captions_generated)}")

    generated_by_image: dict[str, list[str]] = {}
    for entry in captions_generated:
        generated_by_image.setdefault(entry["image_file"], []).append(entry["caption"])

    count_missing = 0
    count_correct = 0

    for cb in captions_balanced:
        image_file = cb["image_file"]
        correct_caption = cb["candidates"][cb["correct_index"]]

        if image_file not in generated_by_image:
            print(f"Not found: {cb}")
            count_missing += 1
            continue

        if correct_caption in generated_by_image[image_file]:
            count_correct += 1
        else:
            print(
                f"Wrong answer for image {image_file}:\n"
                f"Expected: {correct_caption}\n"
                f"Generated: {generated_by_image[image_file]}"
            )

    print(f"Number of missing images: {count_missing}")
    print(f"Number of correct caption matches: {count_correct} of {len(captions_balanced)}")


if __name__ == "__main__":
    validate_json()
