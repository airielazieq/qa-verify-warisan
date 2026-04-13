# Prompt Examples

This folder contains alternative prompt variations that you can use or adapt.

## Available Examples

### 1. paraphrase_technical.txt
**Purpose:** Better for technical/programming content
**Differences from default:**
- Emphasizes preserving technical terminology
- Requests implementation details
- Includes example requirements
- Minimum 60 words (vs 50 in default)

**When to use:**
- Q&A about coding, APIs, frameworks
- Technical documentation
- When precise terminology matters

**How to use:**
```bash
# Copy to main prompts folder
cp prompts/examples/paraphrase_technical.txt prompts/paraphrase_technical.txt

# Update config.yaml
prompts:
  paraphrase: paraphrase_technical.txt
```

---

### 2. language_translate_en.txt
**Purpose:** Translates mixed language to English instead of fixing in-place
**Differs from default:**
- Translates to English instead of normalizing to Malay
- Better for English-speaking teams
- Handles proper nouns and local references

**When to use:**
- Team works primarily in English
- Content needs to be in English
- Mixed BM/Indonesian should become English

**How to use:**
```bash
cp prompts/examples/language_translate_en.txt prompts/language_fix.txt
```

---

### 3. hallucination_strict.txt
**Purpose:** Very strict hallucination detection
**Differs from default:**
- Flags any inference not explicitly in text
- More detailed breakdown
- More conservative scoring

**When to use:**
- Need high accuracy, low false negatives
- Quality is critical
- Should catch even minor hallucinations

**How to use:**
```bash
cp prompts/examples/hallucination_strict.txt prompts/hallucination_check.txt
```

---

## Creating Your Own Variants

1. **Copy an example:**
   ```bash
   cp prompts/examples/paraphrase_technical.txt prompts/paraphrase_custom.txt
   ```

2. **Edit the prompt:**
   - Modify instructions
   - Change output format
   - Adjust tone/style

3. **Register in config.yaml:**
   ```yaml
   prompts:
     paraphrase: paraphrase_custom.txt
     language_fix: language_fix.txt
     hallucination_check: hallucination_check.txt
   ```

4. **Test it:**
   ```bash
   python qa_cleaner.py sample_data.csv -o test_output.csv
   ```

---

## Template Variables Reference

All prompts can use these variables:

| Variable | Used in | Description |
|----------|---------|-------------|
| `{question}` | all | The Q&A question |
| `{answer}` | all | The answer being processed |
| `{chunk}` | all | Reference text/source material |
| `{text}` | language_fix only | Text needing language fixing |

---

## Prompt Engineering Tips

### For better paraphrasing:
- Specify minimum/maximum word count
- Give examples of good outputs
- Specify tone (formal, casual, technical)

### For better language detection:
- Be explicit about target language
- Mention specific language pairs
- Include handling for proper nouns
- Specify name/term preservation rules

### For better hallucination detection:
- Define "hallucination" clearly
- Specify scoring scale meaning
- Give examples of acceptable inferences
- Request structured output format

---

## Sharing Prompts

To share your custom prompts:

1. Save your prompt `.txt` file
2. Add header comments with:
   - Purpose
   - Author
   - Description
   - Use cases
3. Include example config.yaml showing how to use it
4. Test with sample data first

Share via:
- Email attachment
- Shared drive
- Git repository
- Slack/Teams
