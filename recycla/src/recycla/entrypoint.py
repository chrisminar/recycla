import subprocess
from pathlib import Path

import click

from recycla import DATA_PATH, ROOT_PATH, log
from recycla.evaluate.compare_models import compare
from recycla.evaluate.evaluate_test_images import test_image
from recycla.process_data.parse import parse_data
from recycla.train.experiments.models_experiment import model_experiment
from recycla.train.train import train


@click.group()
def recycla():
    pass


recycla.add_command(parse_data)
recycla.add_command(train)
recycla.add_command(compare)
recycla.add_command(model_experiment)


@recycla.command()
def count():
    """
    Count the number of files in the training dataset.
    """
    train_path = DATA_PATH / "labeled_data" / "train"
    try:
        # Try using find command (Unix/Linux systems)
        result = subprocess.run(
            ["find", str(train_path), "-type", "f"],
            capture_output=True,
            text=True,
            check=True,
        )
        file_count = (
            len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0
        )
        click.echo(f"Total files in training dataset: {file_count}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback for systems without find command (like Windows) or when find fails
        file_count = sum(1 for _ in train_path.rglob("*") if _.is_file())
        click.echo(f"Total files in training dataset: {file_count}")


@recycla.command()
@click.argument(
    "file-type", type=click.Choice(["image", "video", "all"], case_sensitive=False)
)
@click.option(
    "--model-path",
    default=ROOT_PATH / ".models/best_candidate.pth",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to the model file.",
)
@click.option(
    "--image-dir",
    default=DATA_PATH / "labeled_data",
    type=click.Path(file_okay=False, path_type=Path),
    help="Path to the image directory.",
)
@click.option(
    "--video-dir",
    default=DATA_PATH / "pi_data/video/2025-03-21_2025-05-15",
    type=click.Path(file_okay=False, path_type=Path),
    help="Path to the image file.",
)
@click.option(
    "--save-path",
    default=ROOT_PATH / ".results",
    type=click.Path(file_okay=False, path_type=Path),
    help="Path to the save directory.",
)
def test(
    file_type: str, model_path: Path, image_dir: Path, video_dir: Path, save_path: Path
):
    """
    Evaluate model accuracy.
    """
    log.setLevel("INFO")
    save_path.mkdir(parents=True, exist_ok=True)
    if file_type in ["image", "all"]:
        log.info("Testing images.")
        test_image(image_dir, model_path, save_path)


if __name__ == "__main__":
    recycla()
