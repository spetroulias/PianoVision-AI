import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
CONFIG_PATH = DATA_DIR / "dataset_config.json"


def download_file(repo_id, revision, remote_path, local_path):
    """Download a dataset file from Hugging Face."""

    local_path.parent.mkdir(parents=True, exist_ok=True)

    cached_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        filename=remote_path
    )

    shutil.copyfile(cached_path, local_path)

    print(f"[READY] {local_path.name}")


def main():

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)

    repo_id = config["repo_id"]
    revision = config["revision"]
    recordings = config["recordings"]

    print(f"Dataset: {repo_id}")
    print(f"Revision: {revision}")
    print(f"Recordings: {len(recordings)}")

    # Download dataset metadata
    download_file(
        repo_id,
        revision,
        "metadata.json",
        DATA_DIR / "metadata.json"
    )

    # Download matching MP4 and MIDI files
    for index, recording in enumerate(recordings, start=1):

        recording_id = recording["id"]

        print(f"\n[{index}/{len(recordings)}] {recording_id}")

        download_file(
            repo_id,
            revision,
            f"Video/{recording_id}.mp4",
            DATA_DIR / "videos" / f"{recording_id}.mp4"
        )

        download_file(
            repo_id,
            revision,
            f"MIDI/{recording_id}.mid",
            DATA_DIR / "reference" / f"{recording_id}.mid"
        )

    print("\nDataset download completed successfully.")


if __name__ == "__main__":
    main()