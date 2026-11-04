from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from PIL import Image


def main():
    repo = Path(__file__).resolve().parents[1]
    with TemporaryDirectory() as directory:
        root = Path(directory)
        for label, color in (("cats", (120, 20, 30)), ("dogs", (30, 20, 120))):
            folder = root / label
            folder.mkdir()
            for index in range(5):
                Image.new("RGB", (224, 224), tuple(channel + index for channel in color)).save(
                    folder / f"{index}.png"
                )
        weights = root / "smoke.pth"
        subprocess.run(
            [sys.executable, "train.py", "--data-dir", str(root), "--gpu_id", "-1",
             "--n_epochs", "1", "--batch_size", "3", "--no-pretrained", "--model_fn", str(weights)],
            cwd=repo, check=True,
        )
        prediction = subprocess.check_output(
            [sys.executable, "predict.py", str(root / "cats" / "0.png"), "--weights", str(weights)],
            cwd=repo, text=True,
        ).strip()
        label, separator, score = prediction.partition(" (dog score: ")
        if label not in {"cat", "dog"} or not separator or not 0 <= float(score.rstrip(")")) <= 1:
            raise ValueError(f"Unexpected prediction: {prediction}")
        print(f"CPU training, checkpoint loading, and image inference passed: {prediction}")


if __name__ == "__main__":
    main()
