#!/usr/bin/env python3
"""
QA Data Cleaner and Validator - Unified Prompt Version
Uses a single unified Ollama prompt for all validation and cleaning tasks.
Processes Q&A datasets to detect and fix issues in answers.
"""

import csv
import argparse
import sys
import requests
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
import math

import pandas as pd
import numpy as np
from prompt_manager import PromptManager

try:
    from sentence_transformers import SentenceTransformer, util
    SEMANTIC_ENCODER = SentenceTransformer('all-MiniLM-L6-v2')
except ImportError:
    print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers", file=sys.stderr)
    SEMANTIC_ENCODER = None


@dataclass
class QARecord:
    """Represents a single QA record with validation results."""
    question: str
    answer: str
    chunk: str
    similarity_score: float = 0.0
    has_mixed_language: bool = False
    has_noise: bool = False
    noise_percentage: float = 0.0
    is_too_short: bool = False
    cleaned_answer: str = ""


class QAValidator:
    """Validates and cleans QA data using a unified Ollama prompt."""

    def __init__(self, prompts_dir: str = "prompts", ollama_host: str = "http://localhost:11434", model: str = "qwen3:8b"):
        """
        Initialize validator with Ollama client and unified prompt.

        Args:
            prompts_dir: Directory containing prompt files
            ollama_host: Ollama server URL
            model: Ollama model name
        """
        self.ollama_host = ollama_host
        self.model = model
        self.prompt_manager = PromptManager(prompts_dir=prompts_dir)
        self.results: List[QARecord] = []
        self.semantic_encoder = SEMANTIC_ENCODER

        # Verify Ollama connection
        self._verify_ollama_connection()

    def _verify_ollama_connection(self):
        """Verify that Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"Connected to Ollama at {self.ollama_host}")
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

    def _calculate_chunk_size_factor(self, chunk: str, answer: str) -> float:
        """
        Calculate adjustment factor for similarity threshold based on chunk size.
        Larger chunks have diluted embeddings, so we lower the threshold.
        
        Args:
            chunk: The reference text chunk
            answer: The answer text
            
        Returns:
            Adjustment factor [0.5, 1.0] to multiply with base threshold
        """
        chunk_len = len(chunk.split())
        answer_len = len(answer.split())
        
        # Ratio of chunk to answer size
        size_ratio = chunk_len / max(answer_len, 1)
        
        # For every 10x increase in size, reduce threshold by ~10%
        # Use logarithmic scaling to handle exponential growth
        adjustment = 1.0 / (1.0 + (math.log10(max(size_ratio, 1)) * 0.1))
        
        # Clamp between 0.5 and 1.0
        return max(0.5, min(1.0, adjustment))
    
    def _detect_hallucination_ssun(self, question: str, answer: str, chunk: str) -> float:
        """
        SSUN Method: Semantic Similarity Understanding Network.
        Uses Sentence Transformers embeddings + cosine similarity.
        Returns hallucination risk score [0, 1].
        
        Args:
            question: The Q&A question (for context)
            answer: The provided answer
            chunk: The reference chunk
            
        Returns:
            Hallucination risk [0.0 = safe, 1.0 = definite hallucination]
        """
        if self.semantic_encoder is None:
            self._debug_log("Semantic encoder not available, returning neutral score")
            return 0.5
        
        try:
            # Step 1: Encode both texts to semantic embeddings
            answer_embedding = self.semantic_encoder.encode(answer, convert_to_tensor=True)
            chunk_embedding = self.semantic_encoder.encode(chunk, convert_to_tensor=True)
            
            # Step 2: Calculate cosine similarity
            semantic_similarity = util.pytorch_cos_sim(answer_embedding, chunk_embedding).item()
            
            # Step 3: Adjust threshold based on chunk size
            size_factor = self._calculate_chunk_size_factor(chunk, answer)
            adjusted_similarity = semantic_similarity * size_factor
            
            # Convert to hallucination risk (inverse of similarity)
            hallucination_risk = 1.0 - adjusted_similarity
            
            self._debug_log(
                f"SSUN: semantic_sim={semantic_similarity:.3f}, "
                f"size_factor={size_factor:.3f}, "
                f"adjusted_sim={adjusted_similarity:.3f}, "
                f"risk={hallucination_risk:.3f}"
            )
            
            return hallucination_risk
        
        except Exception as e:
            self._debug_log(f"SSUN encoding failed: {e}")
            return 0.5

    def _parse_ollama_response(self, response_text: str) -> Dict:
        """Parse the key:value response from Ollama."""
        result = {}

        try:
            lines = response_text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    result[key] = value
        except Exception as e:
            print(f"Parse error: {e}", file=sys.stderr)

        # Fallback: return empty dict with defaults
        return result if result else {"STATUS": "error", "CLEANED_ANSWER": ""}
    
    def _debug_log(self, msg: str):
        """Log debug message to stderr."""
        print(f"[DEBUG] {msg}", file=sys.stderr)

    def _call_ollama(self, prompt: str, timeout: int = 300) -> str:
        """Call Ollama with a prompt and return response."""
        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.3,
                },
                timeout=timeout
            )
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                return ""
        except Exception as e:
            print(f"Ollama call error: {e}", file=sys.stderr)
            return ""

    def _validate_all(self, question: str, answer: str, chunk: str) -> tuple:
        """
        Validate answer on all dimensions in a single Ollama call.
        Returns (is_too_short, has_noise, noise_percentage, has_mixed_language)
        """
        word_count = len(answer.split())
        
        prompt = self.prompt_manager.format_prompt(
            'validate_all',
            question=question,
            answer=answer,
            chunk=chunk,
            word_count=word_count
        )
        response = self._call_ollama(prompt)
        
        # Initialize defaults
        is_too_short = word_count < 10
        has_noise = False
        noise_percentage = 0.0
        has_mixed_language = False
        
        try:
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if 'is_too_short:' in line.lower():
                    is_too_short = 'true' in line.lower()
                elif 'has_noise:' in line.lower():
                    has_noise = 'true' in line.lower()
                elif 'noise_percentage:' in line.lower():
                    value = line.split(':', 1)[1].strip()
                    noise_percentage = float(value)
                elif 'has_mixed_language:' in line.lower():
                    has_mixed_language = 'true' in line.lower()
        except Exception as e:
            print(f"Validation parsing error: {e}", file=sys.stderr)
        
        return is_too_short, has_noise, noise_percentage, has_mixed_language

    def process_record(self, question: str, answer: str, chunk: str) -> QARecord:
        """Process a single QA record with unified prompt and validation."""
        record = QARecord(
            question=question,
            answer=answer,
            chunk=chunk
        )

        try:
            # Compute word length of ORIGINAL answer
            word_length = len(answer.split())

            # Run all validations on the ORIGINAL answer before cleaning
            is_too_short, has_noise, noise_percentage, has_mixed_language = self._validate_all(
                question, answer, chunk
            )
            record.is_too_short = is_too_short
            record.has_noise = has_noise
            record.noise_percentage = noise_percentage
            record.has_mixed_language = has_mixed_language

            # SSUN: Calculate semantic similarity with chunk-size adjustment
            record.similarity_score = self._detect_hallucination_ssun(question, answer, chunk)

            # Call unified prompt to Ollama for cleaning
            prompt = self.prompt_manager.format_prompt(
                'unified_cleaner',
                question=question,
                answer=answer,
                chunk=chunk,
                length=word_length
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
                response_text = result.get('response', '').strip()

                # Debug: log raw response (full)
                self._debug_log(f"\n=== RECORD START ===")
                self._debug_log(f"Question: {question[:80]}...")
                self._debug_log(f"Answer: {answer[:80]}...")
                self._debug_log(f"Raw Ollama Response:\n{response_text}\n")

                # Parse key:value response
                parsed = self._parse_ollama_response(response_text)

                # Debug: log parsed response
                self._debug_log(f"Parsed: {parsed}\n")

                # Extract values safely
                answer_cleaned = parsed.get('CLEANED_ANSWER', answer)

                # Debug: log similarity and status
                self._debug_log(f"SSUN Similarity Score: {record.similarity_score:.3f}")
                self._debug_log(f"=== RECORD END ===\n")

                record.cleaned_answer = str(answer_cleaned).strip() if answer_cleaned else answer

            else:
                print(f"Ollama error: {response.status_code}", file=sys.stderr)
                record.cleaned_answer = answer

        except requests.exceptions.Timeout:
            print(f"Error: Request to Ollama timed out for record", file=sys.stderr)
            record.cleaned_answer = answer
        except Exception as e:
            print(f"Error processing record: {e}", file=sys.stderr)
            record.cleaned_answer = answer

        self.results.append(record)
        return record

    def process_csv(self, input_path: str) -> List[QARecord]:
        """Process entire CSV file."""
        # Try common delimiters in order
        delimiters = [',', '\t', ';', '|']
        df = None

        for delimiter in delimiters:
            try:
                df = pd.read_csv(input_path, delimiter=delimiter, on_bad_lines='skip')
                # Check if this delimiter worked by looking for our expected columns
                df.columns = df.columns.str.lower().str.strip()
                col_matches = sum(1 for col in df.columns if any(x in col for x in ['soalan', 'jawapan', 'potongan']))
                if col_matches >= 2:  # Found at least 2 of our columns
                    print(f"Using delimiter: {repr(delimiter)}")
                    break
                df = None
            except:
                df = None

        if df is None:
            raise ValueError(f"Could not parse CSV file with any common delimiter")

        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()

        # Debug: print actual column names
        print(f"DEBUG: Detected columns: {list(df.columns)}", file=sys.stderr)

        # Map variations of column names
        col_mapping = {}
        for col in df.columns:
            col_clean = col.lower().strip().replace(' ', '_')
            if 'soalan' in col_clean or 'question' in col_clean:
                col_mapping['soalan'] = col
            elif 'jawapan' in col_clean or 'answer' in col_clean:
                col_mapping['jawapan'] = col
            elif 'potongan' in col_clean or 'chunk' in col_clean or 'text' in col_clean:
                col_mapping['potongan_teks'] = col

        required_cols = ['soalan', 'jawapan', 'potongan_teks']
        missing_cols = [col for col in required_cols if col not in col_mapping]

        if missing_cols:
            print(f"DEBUG: Column mapping found: {col_mapping}", file=sys.stderr)
            raise ValueError(f"Missing required columns: {missing_cols}. Found: {list(df.columns)}")

        print(f"Processing {len(df)} records...")

        for idx, row in df.iterrows():
            if idx % 10 == 0:
                print(f"  Progress: {idx + 1}/{len(df)}")

            self.process_record(
                question=str(row[col_mapping['soalan']]).strip(),
                answer=str(row[col_mapping['jawapan']]).strip(),
                chunk=str(row[col_mapping['potongan_teks']]).strip()
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
                'has_mixed_language': record.has_mixed_language
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
        print(f"Average hallucination risk: {df['similarity_score'].mean():.3f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="QA Data Cleaner - Single unified prompt for all validations"
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
        help="Ollama model name (default: qwen3:8b)",
        default="qwen3:8b"
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
