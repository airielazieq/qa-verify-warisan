#!/usr/bin/env python3
"""Clean up unnecessary files from the repository."""

import os
from pathlib import Path

os.chdir(r'c:\Users\PC04\Desktop\qa-verify')

files_to_delete = [
    'ADJUSTMENT_VISUALIZATION.md',
    'ALGORITHM_COMPARISON.md',
    'BEFORE_AFTER_COMPARISON.md',
    'CHANGES_SUMMARY.txt',
    'CHUNK_SIZE_ADJUSTMENT.md',
    'DOCUMENTATION_INDEX.md',
    'IMPLEMENTATION_SUMMARY.md',
    'OLLAMA_SETUP.txt',
    'QUICK_START.md',
    'REFERENCE_CARD.md',
    'SETUP_COMPLETE.txt',
    'SOLUTION_COMPLETE.md',
    'SOLUTION_OVERVIEW.md',
    'SSUN_UPDATE_SUMMARY.md',
    'START_HERE.md',
    'debug_output.txt',
    'algo-similarity.pdf'
]

print("Deleting unnecessary files...")
for f in files_to_delete:
    try:
        os.remove(f)
        print(f'✓ Deleted: {f}')
    except FileNotFoundError:
        pass

print('\n✓ Cleanup complete!')
print('\nRemaining files:')
for item in sorted(os.listdir('.')):
    if not item.startswith('.'):
        print(f'  {item}')
