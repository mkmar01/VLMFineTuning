import json
from pathlib import Path

import fire
from tqdm import tqdm

from generate_captions import generate_caption

DEFAULT_OUTPUTS = {
    "train": "train_captions.json",
    "valid": "valid_captions.json",
}


def build_dataset(root_dir="../data", split="train", output_name=None):
    """
    Build caption dataset for a specific SuperTuxKart split.

    Args:
        root_dir: Base directory containing data splits.
        split: Split to process (e.g., 'train', 'valid').
        output_name: Optional override for the output json filename.
    """
    root = Path(root_dir)
    split_dir = root / split

    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory {split_dir} does not exist.")

    filename = output_name or DEFAULT_OUTPUTS.get(split, f"{split}_captions.json")
    output_path = split_dir / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, str]] = []

    info_files = sorted(split_dir.glob("*_info.json"))
    print(f"Found {len(info_files)} {split} sequences.")

    for info_path in tqdm(info_files):
        base_name = info_path.stem.replace("_info", "")

        for view_index in range(10):
            image_path = split_dir / f"{base_name}_{view_index:02d}_im.jpg"
            if not image_path.exists():
                continue

            captions = generate_caption(str(info_path), view_index)
            for caption in captions:
                entries.append(
                    {
                        "image_file": f"{split}/{image_path.name}",
                        "caption": caption,
                    }
                )

    with output_path.open("w") as f:
        json.dump(entries, f, indent=2)

    print(f"\n✓ Saved {len(entries)} captions to {output_path}")


def main():
    fire.Fire({"build": build_dataset})


if __name__ == "__main__":
    main()
