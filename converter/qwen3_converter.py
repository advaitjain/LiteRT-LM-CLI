"""Qwen3 model converter implementation"""

import sys
from pathlib import Path
from typing import List, Dict, Any
import glob

from .base import ModelConverter
from .download import download_model_files, validate_downloaded_files, get_hf_token


class Qwen3Converter(ModelConverter):
    """Converter for Qwen3 models (0.6B, 1.7B, 4B)"""

    # Required files for Qwen3 models
    REQUIRED_FILES = [
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt"
    ]

    # Model size to build function mapping
    MODEL_BUILDERS = {
        "0.6b": "build_0_6b_model",
        "1.7b": "build_1_7b_model",
        "4b": "build_4b_model"
    }

    def get_required_files(self) -> List[str]:
        """Return list of files to download from HuggingFace"""
        return self.REQUIRED_FILES

    def build_pytorch_model(self, kv_cache_max_len: int = 1024) -> Any:
        """Build the PyTorch model from the downloaded checkpoint

        Args:
            kv_cache_max_len: Maximum KV cache length (used as mask_cache_size)
        """
        try:
            # Import qwen3 model builder from ai_edge_torch
            from ai_edge_torch.generative.examples.qwen import qwen3

            # Get the appropriate builder function for this model size
            if self.model_size not in self.MODEL_BUILDERS:
                raise ValueError(
                    f"Unsupported model size: {self.model_size}. "
                    f"Supported sizes: {list(self.MODEL_BUILDERS.keys())}"
                )

            builder_name = self.MODEL_BUILDERS[self.model_size]
            builder_func = getattr(qwen3, builder_name)

            print(f"Building PyTorch model using qwen3.{builder_name}()...")
            print(f"  mask_cache_size: {kv_cache_max_len}")

            # Build with mask_cache_size for models that use mask_as_input=True
            # When mask_as_input=True, mask_cache_size should be 0
            # When mask_as_input=False, mask_cache_size should be kv_cache_max_len
            # For Qwen3, we use mask_as_input=True, so mask_cache_size=0
            model = builder_func(
                checkpoint_path=str(self.checkpoint_dir),
                custom_loader=None,
                mask_cache_size=0  # 0 because we use mask_as_input=True
            )
            print("✓ PyTorch model built successfully\n")

            return model

        except ImportError as e:
            raise Exception(
                f"Failed to import qwen3 module from ai_edge_torch. "
                f"Make sure ai-edge-torch-nightly is properly installed.\n"
                f"Error: {e}"
            ) from e
        except Exception as e:
            raise Exception(f"Failed to build PyTorch model: {e}") from e

    def get_conversion_config(self) -> Dict[str, Any]:
        """Get Qwen3-specific conversion configuration"""
        # Import required constants from ai_edge_torch
        from ai_edge_torch.generative.layers.kv_cache import KV_LAYOUT_TRANSPOSED

        return {
            "kvcache_layout": KV_LAYOUT_TRANSPOSED,
            "mask_as_input": True
        }

    def convert(
        self,
        quantize: str = "dynamic_int8",
        prefill_seq_len: int = 256,
        kv_cache_max_len: int = 1024,
        output_dir: Path = None
    ) -> Path:
        """Execute the full conversion process for Qwen3 model"""

        print(f"Converting {self.model_name}...")
        print(f"Model size: {self.model_size}")
        print(f"Quantization: {quantize}")
        print(f"Prefill sequence length: {prefill_seq_len}")
        print(f"KV cache max length: {kv_cache_max_len}")

        # Use custom output dir if provided, otherwise use default
        if output_dir:
            self.output_dir = Path(output_dir)
            self.checkpoint_dir = self.output_dir / "checkpoint"

        print(f"Output directory: {self.output_dir}\n")

        # Step 1: Create output directories
        print("[1/4] Creating output directories...")
        self.create_output_directories()
        print(f"✓ Created {self.output_dir}\n")

        # Step 2: Download model files from HuggingFace
        print("[2/4] Downloading model files from HuggingFace...")
        token = get_hf_token()
        if token:
            print("Using HF_TOKEN from environment for authentication")

        try:
            download_model_files(
                repo_id=self.model_name,
                files=self.get_required_files(),
                local_dir=self.checkpoint_dir,
                token=token
            )
        except Exception as e:
            raise Exception(f"Download failed: {e}") from e

        # Validate all files are present
        if not validate_downloaded_files(self.checkpoint_dir, self.get_required_files()):
            raise Exception("Not all required files were downloaded successfully")

        # Step 3: Build PyTorch model
        print("[3/4] Building PyTorch model...")
        model = self.build_pytorch_model(kv_cache_max_len=kv_cache_max_len)

        # Step 4: Convert to TFLite and package to LiteRT-LM format
        print("[4/4] Converting to TFLite and packaging to LiteRT-LM format...")
        print("This may take several minutes...\n")

        try:
            # Import conversion utilities from ai_edge_torch
            from ai_edge_torch.generative.utilities import converter
            from ai_edge_torch.generative.utilities.export_config import ExportConfig

            # Get conversion config
            config_dict = self.get_conversion_config()

            # Create ExportConfig
            export_config = ExportConfig(
                kvcache_layout=config_dict["kvcache_layout"],
                mask_as_input=config_dict["mask_as_input"]
            )

            # Convert to LiteRT-LM format
            # For HF tokenizers, pass the tokenizer.json file path
            tokenizer_path = self.checkpoint_dir / "tokenizer.json"

            # Qwen3 uses <|im_end|> as the stop token
            # Based on tokenizer_config.json and litert-community reference:
            # - <|im_end|>: 151645 (eos_token, the stop token)
            # Note: <|endoftext|> (151643) is the pad_token but should NOT be used as stop token
            stop_token_ids = [151645]

            # Set up prompt templates matching litert-community model
            # These define how to format user/assistant/system messages
            user_prompt_prefix = "<|im_start|>user\n"
            user_prompt_suffix = "<|im_end|>\n"
            model_prompt_prefix = "<|im_start|>assistant\n"
            model_prompt_suffix = "<|im_end|>\n"

            # Use a simpler jinja template that matches litert-community
            # This is compatible with the prompt prefixes/suffixes above
            jinja_prompt_template = (
                "{%- for message in messages -%}"
                "{%- if message.content is string -%}"
                "{%- if message.role == 'user' %}<|im_start|>user\n{{ message.content }}<|im_end|>\n{% endif -%}"
                "{%- if message.role == 'model' %}<|im_start|>assistant\n{{ message.content }}<|im_end|>\n{% endif -%}"
                "{%- if message.role == 'system' %}<|im_start|>system\n{{ message.content }}<|im_end|>\n{% endif -%}"
                "{%- else -%}"
                "{%- if message.role == 'user' %}<|im_start|>user\n"
                "{% elif message.role == 'model' %}<|im_start|>assistant\n"
                "{% elif message.role == 'system' %}<|im_start|>system\n{% endif -%}"
                "{%- for item in message.content %}"
                "{%- if item.type == 'text' %}{{ item.text }}"
                "{% elif item.type == 'image' -%}{{ '<start_of_image>' }}{%- elif item.type == 'audio' -%}{{ '<start_of_audio>' }}{%- endif -%}"
                "{%- endfor -%}"
                "{%- if message.role == 'user' %}<|im_end|>\n"
                "{% elif message.role == 'model' %}<|im_end|>\n"
                "{% elif message.role == 'system' %}<|im_end|>\n{% endif -%}"
                "{%- endif -%}"
                "{%- endfor -%}"
                "{%- if add_generation_prompt %}<|im_start|>assistant\n{% endif -%}"
            )

            # Path to base metadata with sampler params
            base_metadata_path = self.base_dir / "converter" / "qwen3_base_metadata.textproto"

            print(f"Converting with configuration:")
            print(f"  - Output format: litertlm")
            print(f"  - Quantization: {quantize}")
            print(f"  - KV cache layout: transposed")
            print(f"  - Attention mask: as input")
            print(f"  - HF tokenizer path: {tokenizer_path}")
            print(f"  - Stop tokens: {stop_token_ids} (<|im_end|>)")
            print(f"  - Prompt templates: user/model/system with im_start/im_end")
            print(f"  - Chat template: Simple template matching litert-community")
            print(f"  - Sampler: temperature=0.6, top_p=0.95, top_k=20 (from base metadata)\n")

            # Convert to LiteRT-LM format
            # Use multiple prefill sequence lengths like the reference conversion
            # This creates multiple prefill signatures for different input lengths
            if isinstance(prefill_seq_len, int):
                # Convert single value to list of standard prefill lengths
                prefill_seq_lens = [8, 64, 128, 256, 512, 1024]
                # Filter to only include lengths <= kv_cache_max_len
                prefill_seq_lens = [s for s in prefill_seq_lens if s <= kv_cache_max_len]
            else:
                prefill_seq_lens = prefill_seq_len

            print(f"  Using prefill_seq_lens: {prefill_seq_lens}")

            converter.convert_to_litert(
                pytorch_model=model,
                output_path=str(self.output_dir),
                output_name_prefix=self.short_model_name,
                prefill_seq_len=prefill_seq_lens,  # Use list instead of single value
                kv_cache_max_len=kv_cache_max_len,
                quantize=quantize,
                export_config=export_config,
                output_format="litertlm",
                hf_tokenizer_model_path=str(tokenizer_path),
                stop_token_ids=stop_token_ids,
                user_prompt_prefix=user_prompt_prefix,
                user_prompt_suffix=user_prompt_suffix,
                model_prompt_prefix=model_prompt_prefix,
                model_prompt_suffix=model_prompt_suffix,
                jinja_prompt_template=jinja_prompt_template,
                base_llm_metadata_path=str(base_metadata_path)
            )

            print("\n✓ Conversion successful!")

        except Exception as e:
            raise Exception(f"Conversion failed: {e}") from e

        # Step 5: Find and rename the output file to match expected name
        # ai-edge-torch outputs files like: Qwen3-0.6B_q8_ekv1024.litertlm
        # We need to rename it to: Qwen3-0.6B.litertlm
        expected_filename = self.get_expected_output_filename()
        expected_path = self.output_dir / expected_filename

        # Find the generated .litertlm file
        litertlm_files = list(self.output_dir.glob(f"{self.short_model_name}*.litertlm"))

        if len(litertlm_files) == 0:
            raise Exception(f"No .litertlm file found in {self.output_dir}")

        if len(litertlm_files) > 1:
            # If multiple files exist, try to find the most recent one
            litertlm_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        generated_file = litertlm_files[0]

        # Rename if necessary
        if generated_file != expected_path:
            print(f"\nRenaming output file:")
            print(f"  From: {generated_file.name}")
            print(f"  To:   {expected_filename}")
            generated_file.rename(expected_path)

        # Get file size
        file_size_mb = expected_path.stat().st_size / (1024 * 1024)

        print(f"\n{'='*60}")
        print("✓ Model successfully converted!")
        print(f"{'='*60}")
        print(f"Output: {expected_path}")
        print(f"Size: {file_size_mb:.1f} MB")
        print(f"\nYou can now run: litert-lm-cli run {self.model_name}")
        print(f"{'='*60}\n")

        return expected_path
