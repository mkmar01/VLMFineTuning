import json
from pathlib import Path

import fire
from tqdm import tqdm

from generate_qa import generate_qa_pairs  # your own module


def build_dataset(root_dir="../data", split="train", output_name="balanced_qa_pairs.json"):
    """
    Build QA dataset for a particular SuperTuxKart split.

    Args:
        root_dir: Base data directory (defaults to ../data relative to this file)
        split: Dataset split to parse (e.g. 'train', 'valid')
        output_name: Name of the json file to create inside the split directory
    """
    root = Path(root_dir)
    split_dir = root / split

    output_path = split_dir / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Storage for full JSON array
    full_dataset = []

    # Find all *_info.json files
    info_files = sorted(split_dir.glob("*_info.json"))
    print(f"Found {len(info_files)} {split} sequences.")

    for info_path in tqdm(info_files):
        base_name = info_path.stem.replace("_info", "")

        # Loop over all 10 camera views
        for view_index in range(10):
            img_path = split_dir / f"{base_name}_{view_index:02d}_im.jpg"
            if not img_path.exists():
                continue

            # Generate Q/A pairs
            qa_pairs = generate_qa_pairs(str(info_path), view_index)

            # Append formatted entries
            for qa in qa_pairs:
                full_dataset.append(
                    {
                        "question": qa["question"],
                        "answer": qa["answer"],
                        "image_file": f"{split}/{img_path.name}",
                    }
                )

    # Save as a JSON array (not JSONL)
    with open(output_path, "w") as f:
        json.dump(full_dataset, f, indent=2)

    print(f"\n✓ Saved {len(full_dataset)} Q/A pairs to {output_path}")


if __name__ == "__main__":
    fire.Fire({"build": build_dataset})
