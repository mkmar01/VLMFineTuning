import argparse
import zipfile
from pathlib import Path

BLACKLIST = ["__pycache__", ".pyc", ".ipynb", "checkpoint-", "tensorboard", "events.out.tfevents"]
MAXSIZE_MB = 40

# Only keep the minimal checkpoint artifacts that the grader expects.
# Directory name -> set of allowed filenames anywhere inside that directory.
CHECKPOINT_WHITELIST = {
    "clip_model": {"adapter_config.json", "adapter_model.safetensors", "additional_weights.pt"},
    "vlm_model": {"adapter_config.json", "adapter_model.safetensors"},
}


def should_include(path: Path) -> bool:
    path_str = str(path)
    if any(b in path_str for b in BLACKLIST):
        return False

    for checkpoint_dir, allowed_files in CHECKPOINT_WHITELIST.items():
        if checkpoint_dir in path.parts:
            if path.is_dir():
                # Skip nested checkpoint directories like checkpoint-XXXX entirely
                if any(part.startswith("checkpoint-") for part in path.parts):
                    return False
                return True
            return path.name in allowed_files

    return True


def bundle(homework_dir: str, utid: str):
    """
    Usage: python3 bundle.py homework <utid>
    """
    homework_dir = Path(homework_dir).resolve()
    output_path = Path(__file__).parent / f"{utid}.zip"

    # Get the files from the homework directory
    files = []

    for f in homework_dir.rglob("*"):
        if should_include(f):
            files.append(f)

    print("\n".join(str(f.relative_to(homework_dir)) for f in files))

    # Zip all files, keeping the directory structure
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, homework_dir.stem / f.relative_to(homework_dir))

    output_size_mb = output_path.stat().st_size / 1024 / 1024

    if output_size_mb > MAXSIZE_MB:
        print("Warning: The created zip file is larger than expected!")

    print(f"Submission created: {output_path.resolve()!s} {output_size_mb:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("homework")
    parser.add_argument("utid")

    args = parser.parse_args()

    bundle(args.homework, args.utid)
