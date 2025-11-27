import json
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
qa_balanced = DATA_DIR / "valid_grader" / "balanced_qa_pairs.json"

# Pick the first *_qa_pairs.json in data/train (your generated file)
train_dir = DATA_DIR / "train"
generated_candidates = sorted(train_dir.glob("*_qa_pairs.json"))
if not generated_candidates:
    raise FileNotFoundError(f"No *_qa_pairs.json found under {train_dir}")
qa_generated = generated_candidates[0]


def map_valid_to_train(image_file: str) -> str:
    """
    Convert image paths that start with `valid/` to `train/` so we can compare
    the grader's valid set with the generated train QA file.
    """
    path = Path(image_file)
    if path.parts and path.parts[0] == "valid":
        return str(Path("train") / Path(*path.parts[1:]))
    return image_file


def main():
    qa_b = json.load(open(qa_balanced))
    qa_g = json.load(open(qa_generated))
    print(f"Number of qa_pairs golden: {len(qa_b)}")
    print(f"Number of qa_pairs generated: {len(qa_g)}")
    print(f"Comparing golden {qa_balanced} against generated {qa_generated}")

    # if every qa_pair in qa_b also exists in qa_g otherwise print it out
    count = 0
    correct = 0
    for idx, qa in enumerate(qa_b):
        found = False
        image_path = map_valid_to_train(qa["image_file"])
        for qb in qa_g:
            # print("Comparing: " + str(qb["question"]) + " to " + str(qa["question"]))
            if qb["question"] == qa["question"] and qb["image_file"] == image_path:
                found = True
                if qb["answer"] == qa["answer"]:
                    correct += 1
                else:
                    print(f"Wrong answer: {qb}   \nCorrect answer: {qa['answer']}")
                break
        if not found:
            print("Not found: ", qa)
            count += 1
    print(f"Number of qa_pairs missing: {count}  Correct: {correct}")
    
main()
