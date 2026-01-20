"""HuggingFace model download utilities"""

import os
from pathlib import Path
from typing import List, Optional
from huggingface_hub import hf_hub_download


def download_model_files(
    repo_id: str,
    files: List[str],
    local_dir: Path,
    token: Optional[str] = None
) -> None:
    """Download required files from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID (e.g., "Qwen/Qwen3-0.6B")
        files: List of filenames to download
        local_dir: Local directory to save files
        token: Optional HuggingFace authentication token for gated models

    Raises:
        Exception: If download fails for any file
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(files)} files from {repo_id}...")

    for i, filename in enumerate(files, 1):
        # Check if file already exists
        local_path = local_dir / filename
        if local_path.exists():
            file_size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"  [{i}/{len(files)}] ✓ {filename} (already exists, {file_size_mb:.1f} MB)")
            continue

        try:
            print(f"  [{i}/{len(files)}] Downloading {filename}...", end=" ", flush=True)

            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                token=token
            )

            # Get file size
            file_size_mb = Path(downloaded_path).stat().st_size / (1024 * 1024)
            print(f"✓ ({file_size_mb:.1f} MB)")

        except Exception as e:
            print(f"✗ Failed")
            error_msg = str(e)

            # Check for gated repo error
            if "gated repo" in error_msg.lower() or "access to model" in error_msg.lower():
                raise Exception(
                    f"\n✗ Cannot access gated model {repo_id}.\n"
                    f"\nThis model requires accepting a license agreement.\n"
                    f"Please:\n"
                    f"  1. Visit: https://huggingface.co/{repo_id}\n"
                    f"  2. Click 'Agree and access repository'\n"
                    f"  3. Wait a few minutes for access to be granted\n"
                    f"  4. Try the conversion again\n"
                ) from e
            else:
                raise Exception(f"Failed to download {filename}: {e}") from e

    print("✓ All files downloaded successfully\n")


def validate_downloaded_files(directory: Path, required_files: List[str]) -> bool:
    """Validate that all required files exist in the directory.

    Args:
        directory: Directory to check
        required_files: List of required filenames

    Returns:
        True if all files exist, False otherwise
    """
    directory = Path(directory)
    missing_files = []

    for filename in required_files:
        if not (directory / filename).exists():
            missing_files.append(filename)

    if missing_files:
        print(f"✗ Missing required files: {', '.join(missing_files)}")
        return False

    return True


def get_hf_token() -> Optional[str]:
    """Get HuggingFace token from environment variable or file.

    Checks in order:
    1. HF_TOKEN environment variable
    2. HF_TOKEN file in current directory
    3. HF_TOKEN file in parent directory

    Returns:
        Token string if found, None otherwise
    """
    # Check environment variable first
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    # Check for HF_TOKEN file in current directory
    token_file = Path("HF_TOKEN")
    if token_file.exists():
        try:
            return token_file.read_text().strip()
        except Exception:
            pass

    # Check for HF_TOKEN file in parent directory
    token_file = Path("..") / "HF_TOKEN"
    if token_file.exists():
        try:
            return token_file.read_text().strip()
        except Exception:
            pass

    return None
