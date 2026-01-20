"""Abstract base class for model converters"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any


class ModelConverter(ABC):
    """Abstract base class for converting models to LiteRT-LM format"""

    def __init__(self, model_name: str, base_dir: Path, model_size: str):
        """Initialize the converter.

        Args:
            model_name: HuggingFace model identifier (e.g., "Qwen/Qwen3-0.6B")
            base_dir: Base directory for the CLI tool
            model_size: Model size variant (e.g., "0.6b", "1.7b")
        """
        self.model_name = model_name
        self.base_dir = Path(base_dir)
        self.model_size = model_size

        # Parse model name to determine output paths
        # e.g., "Qwen/Qwen3-0.6B" -> models/Qwen/Qwen3-0.6B/
        parts = model_name.split("/")
        if len(parts) == 2:
            self.org_name, self.short_model_name = parts
        else:
            self.org_name = ""
            self.short_model_name = model_name

        # Set up directory structure
        if self.org_name:
            self.output_dir = self.base_dir / "models" / self.org_name / self.short_model_name
        else:
            self.output_dir = self.base_dir / "models" / self.short_model_name

        self.checkpoint_dir = self.output_dir / "checkpoint"

    @abstractmethod
    def get_required_files(self) -> List[str]:
        """Return list of files to download from HuggingFace.

        Returns:
            List of filenames required for this model
        """
        pass

    @abstractmethod
    def build_pytorch_model(self) -> Any:
        """Build the PyTorch model from the downloaded checkpoint.

        Returns:
            PyTorch model instance
        """
        pass

    @abstractmethod
    def get_conversion_config(self) -> Dict[str, Any]:
        """Get model-specific conversion configuration.

        Returns:
            Dictionary of configuration parameters for ai-edge-torch conversion
        """
        pass

    @abstractmethod
    def convert(
        self,
        quantize: str = "dynamic_int8",
        prefill_seq_len: int = 256,
        kv_cache_max_len: int = 1024,
        output_dir: Path = None
    ) -> Path:
        """Execute the full conversion process.

        Args:
            quantize: Quantization scheme to use
            prefill_seq_len: Prefill sequence length
            kv_cache_max_len: KV cache maximum length
            output_dir: Optional custom output directory

        Returns:
            Path to the generated .litertlm file
        """
        pass

    def create_output_directories(self):
        """Create necessary output directories"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def get_expected_output_filename(self) -> str:
        """Get the expected output filename for the .litertlm file.

        Returns:
            Expected filename (e.g., "Qwen3-0.6B.litertlm")
        """
        return f"{self.short_model_name}.litertlm"
