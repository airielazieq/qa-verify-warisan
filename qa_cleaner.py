#!/usr/bin/env python3
"""
QA Data Cleaner and Validator
Processes Q&A datasets to detect hallucination, mixed languages, noise, and improve answer quality.
Uses Ollama with Qwen 2.5 27B for LLM operations.
Implements SSUN (Semantic Similarity Understanding Network) for hallucination detection.
Based on: "LLM-Based QA from Structured Bibliographic Data via Semantic Similarity Understanding Network"
"""

import csv
import re
import argparse
import sys
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import Counter
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

from prompt_manager import PromptManager

# Configuration
MIN_ANSWER_LENGTH = 20  # Minimum character length for answer
SIMILARITY_THRESHOLD = 0.5  # Threshold for hallucination detection
NOISE_THRESHOLD = 0.3  # Threshold for noise detection (% of non-alphanumeric chars)

# Distinctive language markers for BM and Indonesian.
# These are intentionally domain-agnostic so detection works across diverse topics.
BM_MARKERS = {
    'ialah', 'boleh', 'kerana', 'sahaja', 'sekiranya', 'walaupun',
    'manakala', 'berkenaan', 'mesti', 'daripada'
}

INDONESIAN_MARKERS = {
    'adalah', 'bisa', 'karena', 'saja', 'jika', 'meskipun',
    'sedangkan', 'terkait', 'harus', 'dari'
}


@dataclass
class QARecord:
    """Represents a single QA record with metadata."""
    question: str
    answer: str
    chunk: str
    similarity_score: float = 0.0
    has_mixed_language: bool = False
    mixed_language_details: str = ""
    has_noise: bool = False
    noise_percentage: float = 0.0
    is_too_short: bool = False
    paraphrased_answer: str = ""
    cleaned_answer: str = ""


class QAValidator:
    """Validates and cleans QA data using SSUN-based semantic similarity."""

    def __init__(self, prompts_dir: str = "prompts", ollama_host: str = "http://localhost:11434", model: str = "qwen2.5:27b"):
        """
        Initialize validator with Ollama client, prompt manager, and semantic encoder.

        Args:
            prompts_dir: Directory containing prompt files and config
            ollama_host: Ollama server URL (default: localhost:11434)
            model: Ollama model name (default: qwen2.5:27b)
        """
        self.ollama_host = ollama_host
        self.model = model
        self.prompt_manager = PromptManager(prompts_dir=prompts_dir)
        self.results: List[QARecord] = []

        # Initialize Sentence Transformer for semantic similarity (SSUN algorithm)
        print("Loading Sentence Transformer for semantic similarity...")
        try:
            self.semantic_encoder = SentenceTransformer('all-MiniLM-L6-v2')
            print("Semantic encoder loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load semantic encoder: {e}", file=sys.stderr)
            print("Will use fallback similarity method", file=sys.stderr)
            self.semantic_encoder = None

        # Verify Ollama connection
        self._verify_ollama_connection()

    def _verify_ollama_connection(self):
        """Verify that Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"Connected to Ollama at {self.ollama_host}")
                # List available models
                models = response.json().get('models', [])
                model_names = [m.get('name') for m in models]
                print(f"Available models: {model_names}")

                if not any(self.model in m for m in model_names):
                    print(f"Warning: Model '{self.model}' not found in available models")
                    print(f"Make sure to pull it: ollama pull {self.model}")
            else:
                raise ConnectionError(f"Ollama returned status {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"Error: Cannot connect to Ollama at {self.ollama_host}")
            print("Make sure Ollama is running: ollama serve")
            sys.exit(1)
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            sys.exit(1)

    def detect_hallucination(self, question: str, answer: str, chunk: str) -> float:
        """
        Detect hallucination using SSUN (Semantic Similarity Understanding Network) algorithm.

        Compares answer with chunk using semantic similarity (cosine similarity of embeddings).
        This is more robust than string-based similarity as it captures semantic meaning.

        Based on: "LLM-Based QA from Structured Bibliographic Data via Semantic Similarity Understanding Network"

        Returns:
            Similarity score between 0 and 1 (higher = more likely hallucinated)
        """
        if self.semantic_encoder is not None:
            # SSUN Method: Sentence Transformer + Cosine Similarity
            try:
                # Step 1: Semantic Encoding - Create embeddings for answer and chunk
                answer_embedding = self.semantic_encoder.encode(answer, convert_to_tensor=True)
                chunk_embedding = self.semantic_encoder.encode(chunk, convert_to_tensor=True)

                # Step 2: Semantic Similarity Calculation - Cosine similarity
                # Returns similarity in range [0, 1]
                semantic_similarity = util.pytorch_cos_sim(answer_embedding, chunk_embedding).item()

                # Higher semantic_similarity = answer is similar to chunk = NOT hallucinated
                # We want to return hallucination risk, so invert it
                hallucination_risk = 1.0 - semantic_similarity

                return max(0.0, min(1.0, hallucination_risk))

            except Exception as e:
                print(f"Error in SSUN semantic similarity: {e}", file=sys.stderr)
                # Fall back to basic similarity
                return self._fallback_similarity(question, answer, chunk)
        else:
            # Fallback if semantic encoder not available
            return self._fallback_similarity(question, answer, chunk)

    def _fallback_similarity(self, question: str, answer: str, chunk: str) -> float:
        """
        Fallback similarity method using SequenceMatcher (basic string similarity).
        Used when semantic encoder is not available.

        Returns:
            Similarity score between 0 and 1
        """
        answer_clean = answer.lower()
        chunk_clean = chunk.lower()
        question_clean = question.lower()

        # Calculate similarity between answer and chunk using SequenceMatcher
        string_similarity = SequenceMatcher(None, answer_clean, chunk_clean).ratio()

        # Also check phrase overlap
        chunk_phrases = set(chunk_clean.split())
        answer_phrases = set(answer_clean.split())
        overlap = len(chunk_phrases & answer_phrases) / max(len(answer_phrases), 1)

        # Combined: 40% string similarity + 60% overlap
        combined_score = (string_similarity * 0.4) + (overlap * 0.6)

        # Invert to get hallucination risk (similarity = not hallucinated)
        hallucination_risk = 1.0 - combined_score

        return max(0.0, min(1.0, hallucination_risk))

    def check_answer_length(self, answer: str) -> bool:
        """Check if answer is below minimum threshold."""
        return len(answer.strip()) < MIN_ANSWER_LENGTH

    def detect_noise(self, text: str) -> Tuple[bool, float]:
        """
        Detect noise in text (random characters, excessive punctuation, etc).
        Returns (has_noise, noise_percentage).
        """
        if not text:
            return False, 0.0

        # Count non-alphanumeric characters
        non_alphanum = sum(1 for c in text if not c.isalnum() and c.isascii() and c not in ' ')
        total_chars = len(text)

        noise_ratio = non_alphanum / total_chars if total_chars > 0 else 0

        # Check for repeated characters (noise indicator)
        repeated_chars = len(re.findall(r'(.)\1{3,}', text))  # 4+ consecutive same chars
        repeated_ratio = repeated_chars / max(total_chars, 1)

        combined_noise = (noise_ratio * 0.6) + (repeated_ratio * 0.4)

        has_noise = combined_noise > NOISE_THRESHOLD

        return has_noise, combined_noise

    def detect_mixed_language(self, text: str) -> Tuple[bool, str]:
        """
        Detect if text appears to mix Bahasa Malaysia and Indonesian.
        Returns (has_mixed_language, details_string).
        """
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)

        if len(words) < 5:
            return False, "Text too short for language detection"

        bm_score = sum(1 for word in words if word in BM_MARKERS)
        indo_score = sum(1 for word in words if word in INDONESIAN_MARKERS)
        total_markers = bm_score + indo_score

        if total_markers < 3:
            return False, f"Insufficient language markers (BM: {bm_score}, Indonesian: {indo_score})"

        bm_ratio = bm_score / total_markers
        indo_ratio = indo_score / total_markers
        has_mixed = (
            bm_score >= 1 and
            indo_score >= 1 and
            bm_ratio >= 0.2 and
            indo_ratio >= 0.2
        )

        details = (
            f"BM markers: {bm_score}, Indonesian markers: {indo_score}, "
            f"BM ratio: {bm_ratio:.2f}, Indonesian ratio: {indo_ratio:.2f}"
        )

        return has_mixed, details

    def paraphrase_short_answer(self, question: str, answer: str, chunk: str) -> str:
        """
        Use Ollama to paraphrase a short answer using information from the chunk.
        """
        try:
            prompt = self.prompt_manager.format_prompt(
                'paraphrase',
                question=question,
                answer=answer,
                chunk=chunk
            )

            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.7,
                },
                timeout=300
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                print(f"Ollama error: {response.status_code}", file=sys.stderr)
                return answer
        except requests.exceptions.Timeout:
            print(f"Error: Request to Ollama timed out", file=sys.stderr)
            return answer
        except Exception as e:
            print(f"Error in paraphrasing: {e}", file=sys.stderr)
            return answer

    def check_and_fix_language_mix(self, text: str, question: str, chunk: str) -> str:
        """
        Use Ollama to identify and fix mixed language issues.
        """
        try:
            # Truncate very long texts to avoid token limits
            question = question[:500]
            chunk = chunk[:500]
            text = text[:500]

            prompt = self.prompt_manager.format_prompt(
                'language_fix',
                text=text,
                question=question,
                chunk=chunk
            )

            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.7,
                },
                timeout=300
            )

            if response.status_code == 200:
                result = response.json()
                fixed_text = result.get('response', '').strip()
                return fixed_text if fixed_text else text
            else:
                print(f"Ollama error: {response.status_code}", file=sys.stderr)
                return text
        except requests.exceptions.Timeout:
            print(f"Error: Request to Ollama timed out", file=sys.stderr)
            return text
        except Exception as e:
            print(f"Error in language fixing: {e}", file=sys.stderr)
            return text

    def clean_noise(self, text: str) -> str:
        """Basic noise cleaning (remove excessive punctuation, fix repeated chars)."""
        # Remove multiple consecutive spaces
        text = re.sub(r'\s+', ' ', text)

        # Fix excessive punctuation
        text = re.sub(r'([!?.]){3,}', r'\1', text)

        # Fix repeated characters (but preserve intentional ones like "aaa")
        text = re.sub(r'([a-zA-Z])\1{4,}', r'\1\1', text)

        return text.strip()

    def process_record(self, question: str, answer: str, chunk: str) -> QARecord:
        """Process a single QA record through all validation steps."""
        record = QARecord(
            question=question,
            answer=answer,
            chunk=chunk
        )

        # 1. Detect hallucination
        record.similarity_score = self.detect_hallucination(question, answer, chunk)

        # 2. Check noise
        record.has_noise, record.noise_percentage = self.detect_noise(answer)
        record.cleaned_answer = self.clean_noise(answer) if record.has_noise else answer

        # 3. Check answer length
        record.is_too_short = self.check_answer_length(record.cleaned_answer)

        # 4. Paraphrase if too short
        if record.is_too_short:
            record.paraphrased_answer = self.paraphrase_short_answer(
                question, record.cleaned_answer, chunk
            )
            record.cleaned_answer = record.paraphrased_answer

        # 5. Detect mixed language
        record.has_mixed_language, record.mixed_language_details = self.detect_mixed_language(
            record.cleaned_answer
        )

        # 6. Fix language mix if detected
        if record.has_mixed_language:
            record.cleaned_answer = self.check_and_fix_language_mix(
                record.cleaned_answer,
                question,
                chunk
            )

        self.results.append(record)
        return record

    def process_csv(self, input_path: str) -> List[QARecord]:
        """Process entire CSV file."""
        df = pd.read_csv(input_path)

        # Validate required columns
        required_cols = ['soalan', 'jawapan', 'potongan teks']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        print(f"Processing {len(df)} records...")

        for idx, row in df.iterrows():
            if idx % 10 == 0:
                print(f"  Progress: {idx + 1}/{len(df)}")

            self.process_record(
                question=str(row['soalan']).strip(),
                answer=str(row['jawapan']).strip(),
                chunk=str(row['potongan teks']).strip()
            )

        return self.results

    def export_results(self, output_path: str):
        """Export cleaned results to CSV."""
        data = []

        for record in self.results:
            data.append({
                'soalan': record.question,
                'jawapan_original': record.answer,
                'jawapan_cleaned': record.cleaned_answer,
                'potongan_teks': record.chunk,
                'similarity_score': round(record.similarity_score, 3),
                'has_noise': record.has_noise,
                'noise_percentage': round(record.noise_percentage, 3),
                'is_too_short': record.is_too_short,
                'has_mixed_language': record.has_mixed_language,
                'language_details': record.mixed_language_details
            })

        output_df = pd.DataFrame(data)
        output_df.to_csv(output_path, index=False, encoding='utf-8')

        print(f"\nResults exported to: {output_path}")
        print(f"Total records processed: {len(data)}")

        # Print statistics
        self._print_statistics(output_df)

    def _print_statistics(self, df: pd.DataFrame):
        """Print summary statistics."""
        print("\n=== STATISTICS ===")
        print(f"Records with noise: {df['has_noise'].sum()} ({df['has_noise'].sum()/len(df)*100:.1f}%)")
        print(f"Records too short: {df['is_too_short'].sum()} ({df['is_too_short'].sum()/len(df)*100:.1f}%)")
        print(f"Records with mixed language: {df['has_mixed_language'].sum()} ({df['has_mixed_language'].sum()/len(df)*100:.1f}%)")
        print(f"Average similarity score: {df['similarity_score'].mean():.3f}")
        print(f"High hallucination risk (similarity > {SIMILARITY_THRESHOLD}): {(df['similarity_score'] > SIMILARITY_THRESHOLD).sum()} records")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="QA Data Cleaner - Detects and fixes issues in Q&A datasets using Ollama"
    )
    parser.add_argument(
        "input_file",
        help="Input CSV file path"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output CSV file path (default: input_file_cleaned.csv)",
        default=None
    )
    parser.add_argument(
        "--ollama-host",
        help="Ollama server host (default: http://localhost:11434)",
        default="http://localhost:11434"
    )
    parser.add_argument(
        "-m", "--model",
        help="Ollama model name (default: qwen2.5:27b)",
        default="qwen2.5:27b"
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Set output path
    output_path = args.output or str(input_path.parent / f"{input_path.stem}_cleaned.csv")

    # Process
    try:
        validator = QAValidator(ollama_host=args.ollama_host, model=args.model)
        validator.process_csv(str(input_path))
        validator.export_results(output_path)
        print("\n[OK] Processing complete!")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
