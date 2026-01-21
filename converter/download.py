"""HuggingFace model download utilities"""

import os
import sys
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

            # Check for common auth errors
            is_auth_error = "401" in error_msg or "403" in error_msg or "gated repo" in error_msg.lower() or "access to model" in error_msg.lower()
            
            if is_auth_error:
                print(f"\n✗ Error: Authentication failed for {repo_id}")
                print(f"  This model is likely gated or requires authentication.\n")
                print("  Please authenticate with HuggingFace:")
                print("  1. Run: hf login")
                print("  2. Enter your HuggingFace token (get it from https://huggingface.co/settings/tokens)")
                print("\n  If you've already logged in, ensure your token has access to this model.")
                print(f"  For gated models like {repo_id}, you also need to:")
                print(f"  - Visit https://huggingface.co/{repo_id}")
                print("  - Click 'Agree and access repository'")
                print("  - Wait a few minutes for access to be granted\n")
                
                # Check if we have a token but it might be invalid
                if token:
                    print("  Note: A token was found properly, but access was still denied.")
                    print("  Please check that your token has the 'read' permission.")
                
                sys.exit(1)
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
    2. ~/.cache/huggingface/token (standard huggingface-cli login location)
    3. HF_TOKEN file in current directory
    4. HF_TOKEN file in parent directory

    Returns:
        Token string if found, None otherwise
    """
    # Check environment variable first
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    # Check standard huggingface-cli cache location
    token_path = Path.home() / '.cache' / 'huggingface' / 'token'
    if token_path.exists():
        try:
            return token_path.read_text().strip()
        except Exception:
            pass

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
