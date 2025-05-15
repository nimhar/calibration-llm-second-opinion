# Second Opinions Experiment

This repository contains code for running a context-conditioned second opinion experiment on medical and general knowledge datasets. The experiment tests how AI models respond when providing a second opinion after being given a first answer suggestion.

## Overview

The experiment pipeline evaluates how different contextual factors (such as information source, gender, age, and experience) affect the responses when a model is asked to provide a second opinion on multiple-choice questions.

## Supported Models

- **Azure OpenAI** (GPT-4o)
- **Vertex AI / Gemini**
- **Hugging Face** models (Llama-3.1, DeepSeek)

## Datasets

- **NEJM-AI**: Medical questions from various specialties
- **MMLU**: Multiple-choice questions across diverse domains

## Usage

```bash
# Run experiment with GPT-4o on MMLU dataset
python so_experiment_pipeline.py mmlu gpt --output-dir outputs

# Run experiment with Gemini on NEJM-AI dataset
python so_experiment_pipeline.py nejm gemini --output-dir outputs_nejm_gemini

# Resume a partially completed run
python so_experiment_pipeline.py nejm gpt --resume outputs/nejm/gpt_nejm_2023-08-01_partial.csv
```

## Environment Setup

Required environment variables:
- For Azure OpenAI: `AZURE_API_KEY_4o`, `AZURE_API_VERSION`, `AZURE_ENDPOINT_4o`
- For Gemini: `GCP_PROJECT_ID`, `GCP_LOCATION_PROJECT`
- For HuggingFace: `HUGGINGFACEHUB_API_TOKEN`

## Output

Results are saved as CSV files with detailed information about:
- Original questions and correct answers
- Baseline model responses
- First opinions given to the model
- Second opinions provided by the model
- Contextual factors used in the prompts
- Accuracy metrics
