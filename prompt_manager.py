"""
Prompt Manager - Handles loading and managing prompts from external files
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Optional


class PromptManager:
    """Manages prompts loaded from external files."""

    def __init__(self, prompts_dir: str = "prompts"):
        """
        Initialize prompt manager.

        Args:
            prompts_dir: Directory containing prompt files and config
        """
        self.prompts_dir = Path(prompts_dir)
        self.config_path = self.prompts_dir / "config.yaml"
        self.prompts: Dict[str, str] = {}
        self.config: Dict = {}

        self._load_config()
        self._load_prompts()

    def _load_config(self):
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f) or {}

    def _load_prompts(self):
        """Load all prompts defined in config."""
        prompt_mappings = self.config.get('prompts', {})

        for prompt_name, prompt_file in prompt_mappings.items():
            prompt_path = self.prompts_dir / prompt_file

            if not prompt_path.exists():
                raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.prompts[prompt_name] = f.read().strip()

            print(f"Loaded prompt: {prompt_name}")

    def get_prompt(self, name: str) -> str:
        """
        Get a prompt by name.

        Args:
            name: Prompt name (must be defined in config.yaml)

        Returns:
            Prompt template string

        Raises:
            KeyError: If prompt not found
        """
        if name not in self.prompts:
            available = list(self.prompts.keys())
            raise KeyError(f"Prompt '{name}' not found. Available: {available}")

        return self.prompts[name]

    def format_prompt(self, name: str, **kwargs) -> str:
        """
        Get and format a prompt with provided arguments.

        Args:
            name: Prompt name
            **kwargs: Variables to replace in prompt template

        Returns:
            Formatted prompt string
        """
        prompt = self.get_prompt(name)
        return prompt.format(**kwargs)

    def list_prompts(self) -> Dict[str, str]:
        """Get all loaded prompts."""
        return self.prompts.copy()

    def add_custom_prompt(self, name: str, content: str):
        """
        Add a custom prompt to memory (not saved to file).

        Args:
            name: Prompt name
            content: Prompt template content
        """
        self.prompts[name] = content

    def reload_prompts(self):
        """Reload all prompts from disk."""
        self.prompts.clear()
        self._load_config()
        self._load_prompts()

    def get_metadata(self) -> Dict:
        """Get prompt metadata from config."""
        return self.config.get('metadata', {})
