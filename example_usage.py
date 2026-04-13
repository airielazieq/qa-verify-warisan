#!/usr/bin/env python3
"""
Example: Using QA Cleaner with Ollama
"""

from qa_cleaner import QAValidator

# Initialize validator with Ollama
# Note: Make sure Ollama is running: ollama serve
# And the model is pulled: ollama pull qwen2.5:27b

validator = QAValidator(
    ollama_host="http://localhost:11434",
    model="qwen2.5:27b"
)

# Process a single record
record = validator.process_record(
    question="Bagaimana cara membuat list dalam Python?",
    answer="gunakan []",
    chunk="Untuk membuat list dalam Python, gunakan tanda kurung siku [] dengan elemen-elemen dipisahkan oleh koma. Contoh: my_list = [1, 2, 3, 'a', 'b']"
)

# Check results
print(f"Question: {record.question}")
print(f"Original Answer: {record.answer}")
print(f"Cleaned Answer: {record.cleaned_answer}")
print(f"Hallucination Risk: {record.similarity_score:.2%}")
print(f"Too Short: {record.is_too_short}")
print(f"Has Noise: {record.has_noise}")
print(f"Mixed Language: {record.has_mixed_language}")

# Or process an entire CSV file
print("\n" + "="*50)
print("Processing CSV file with Ollama...")
print("="*50)

results = validator.process_csv("sample_data.csv")
validator.export_results("example_output.csv")

# Access results programmatically
for i, result in enumerate(results[:2], 1):
    print(f"\nRecord {i}:")
    print(f"  Similarity Score: {result.similarity_score:.3f}")
    print(f"  Noise: {result.has_noise} ({result.noise_percentage:.1%})")
    print(f"  Too short: {result.is_too_short}")
    print(f"  Mixed Language: {result.has_mixed_language}")

