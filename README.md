# LLM-Based Multilingual NLP Workflow for Toxicity Detection and Detoxification

This project uses LangChain and Granite-3.0-2B for toxicity detection and detoxification, with Helsinki-NLP for multilingual translation.

## Features
- Translates non-English text to English.
- Detects toxicity and rewrites toxic text politely.
- Outputs results in CSV format.
- Evaluates the annotator agreement on the detoxification.

## Installation
```bash
pip install langchain huggingface_hub transformers langdetect torch pandas sklearn numpy
```

Update HF_API_TOKEN in the code with your Hugging Face API token:

```bash
HF_API_TOKEN = "your_token_here"
```

## Usage
1. **Prepare Datasets**:
   - Add your multilingual data to a CSV file with a `sentence` column.
   - Add your toxicity data to a CSV file with a `text` column.
   - Update the file paths in `main.py`:
     ```python
     multi_text = pd.read_csv('your_multilingual_file.csv')
     toxicity_text = pd.read_csv('your_toxicity_file.csv')
     ```

2. **Run**:
  ```bash
  python main.py
  ```

3. **Output**:
  - Results saved as `multi_df.csv` (multilingual) and `toxic_df.csv` (toxicity).
  - Columns: `text`, `toxicity`, `polite version`.

4. **Evaluation**:
  - Prints toxic samples for manual review.
  - Includes a placeholder for inter-annotator agreement (Cohen's Kappa).

## Notes
- Replace your hugging face token, empty dataset paths and annotator scores in the code.
- GPU recommended for faster processing.

## License

MIT License
