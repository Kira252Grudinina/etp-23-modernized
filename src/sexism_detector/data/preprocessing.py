"""
Text preprocessing module for sexism detection.

Modernized from original 2023 notebook with type hints, caching, and batch processing.
"""

from dataclasses import dataclass
from typing import List, Union
from functools import lru_cache
import re
import emoji
import string
from tqdm.auto import tqdm


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing pipeline."""
    remove_emojis: bool = True
    translate_emojis: str | None = None  # "en" | "alias" | None
    remove_urls: bool = True
    remove_users: bool = True
    lowercase: bool = False  # Keep False for BERT models
    remove_punctuation: bool = False
    min_length: int = 3
    max_length: int = 512
    normalize_whitespace: bool = True


class TextPreprocessor:
    """Modern text preprocessor with caching and batch support."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self._cache_hits = 0
    
    @lru_cache(maxsize=10000)
    def process_single(self, text: str) -> str:
        """Process a single text with caching."""
        if not text or not isinstance(text, str):
            return ""
        
        self._cache_hits += 1
        
        # Remove URL placeholders and actual URLs
        if self.config.remove_urls:
            text = re.sub(r'\[URL\]', '', text)
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mention placeholders and actual mentions
        if self.config.remove_users:
            text = re.sub(r'\[USER\]', '', text)
            text = re.sub(r'@\w+', '', text)
        
        # Handle emojis
        if self.config.remove_emojis:
            text = emoji.replace_emoji(text, '')
        elif self.config.translate_emojis:
            text = emoji.demojize(
                text,
                delimiters=('', ''),
                language=self.config.translate_emojis
            )
            text = text.replace('_', ' ')
        
        # Lowercase if specified
        if self.config.lowercase:
            text = text.lower()
        
        # Remove punctuation if specified
        if self.config.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Normalize whitespace
        if self.config.normalize_whitespace:
            text = ' '.join(text.split())
        
        # Length filtering
        text = text.strip()
        if len(text) < self.config.min_length:
            return ""
        
        if len(text) > self.config.max_length:
            text = text[:self.config.max_length]
        
        return text
    
    def process_batch(self, texts: List[str], show_progress: bool = False) -> List[str]:
        """Process a batch of texts efficiently."""
        iterator = tqdm(texts, desc="Processing") if show_progress else texts
        return [self.process_single(text) for text in iterator]
    
    def __call__(self, examples: Union[dict, List[str], str]) -> Union[dict, List[str], str]:
        """Make preprocessor callable for HuggingFace datasets.map()."""
        if isinstance(examples, dict) and 'text' in examples:
            if isinstance(examples['text'], list):
                examples['text'] = self.process_batch(examples['text'])
            else:
                examples['text'] = self.process_single(examples['text'])
            return examples
        elif isinstance(examples, list):
            return self.process_batch(examples)
        elif isinstance(examples, str):
            return self.process_single(examples)
        else:
            raise ValueError(f"Unsupported input type: {type(examples)}")
    
    def get_cache_info(self) -> dict:
        """Get cache statistics."""
        cache_info = self.process_single.cache_info()
        return {
            "hits": cache_info.hits,
            "misses": cache_info.misses,
            "cache_size": cache_info.currsize,
            "max_cache_size": 10000
        }
    
    def clear_cache(self):
        """Clear the LRU cache."""
        self.process_single.cache_clear()


# Preset configurations
PRESET_CONFIGS = {
    "minimal": PreprocessingConfig(
        remove_emojis=True,
        remove_urls=True,
        remove_users=True,
    ),
    "bert_optimized": PreprocessingConfig(
        remove_emojis=True,
        remove_urls=True,
        remove_users=True,
        lowercase=False,
        remove_punctuation=False,
    ),
}


def get_preprocessor(preset: str = "bert_optimized") -> TextPreprocessor:
    """Factory function to create preprocessor with preset config."""
    if preset not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(PRESET_CONFIGS.keys())}")
    return TextPreprocessor(PRESET_CONFIGS[preset])
