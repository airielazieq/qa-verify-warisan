#!/usr/bin/env python3
"""
Example: Using QA Cleaner with Ollama
"""

from qa_cleaner import QAValidator

# Make sure Ollama is running:  ollama serve
# And the model is pulled:      ollama pull qwen3:8b
#
# For maximum throughput, set OLLAMA_NUM_PARALLEL before starting Ollama:
#   $env:OLLAMA_NUM_PARALLEL = "4"   (PowerShell)
#   set OLLAMA_NUM_PARALLEL=4        (cmd)

validator = QAValidator(
    ollama_host="http://localhost:11434",
    model="qwen3:8b",
    workers=4,          # concurrent Ollama requests — match OLLAMA_NUM_PARALLEL
    context_size=32768, # large enough for 6000-word chunks
)

# Process an entire CSV file (batch SSUN + parallel LLM calls)
results = validator.process_csv("input.csv")
validator.export_results("example_output.csv")

# Access results programmatically
for i, r in enumerate(results[:2], 1):
    print(f"\nRecord {i}:")
    print(f"  Question:         {r.question[:60]}…")
    print(f"  Original Answer:  {r.answer[:60]}…")
    print(f"  Cleaned Answer:   {r.cleaned_answer[:60]}…")
    print(f"  Hallucination Risk: {r.similarity_score:.2%}")
    print(f"  Too Short:        {r.is_too_short}")
    print(f"  Has Noise:        {r.has_noise} ({r.noise_percentage:.1%})")
    print(f"  Mixed Language:   {r.has_mixed_language}")
