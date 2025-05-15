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

## Pipeline

The project consists of two main components:

1. **Experiment Pipeline** (`so_experiment_pipeline.py`): Runs the model queries with contextual prompts
2. **Evaluation Pipeline** (`evaluation_pipeline.py`): Analyzes results and generates calibration metrics

## Usage

### Running Experiments

```bash
# Run experiment with GPT-4o on MMLU dataset
python so_experiment_pipeline.py mmlu gpt --output-dir outputs

# Run experiment with Gemini on NEJM-AI dataset
python so_experiment_pipeline.py nejm gemini --output-dir outputs_nejm_gemini

# Resume a partially completed run
python so_experiment_pipeline.py nejm gpt --resume outputs/nejm/gpt_nejm_2023-08-01_partial.csv
```

### Running Evaluation

```bash
# Analyze experiment results
python -m evaluation_pipeline --csv_paths path/to/results.csv --output analysis_output

# Run specific evaluation components
python -m evaluation_pipeline --csv_paths results.csv --output analysis_output --do-primary-evaluation --do-sampling
```

## Environment Setup

Required environment variables:
- For Azure OpenAI: `AZURE_API_KEY_4o`, `AZURE_API_VERSION`, `AZURE_ENDPOINT_4o`
- For Gemini: `GCP_PROJECT_ID`, `GCP_LOCATION_PROJECT`
- For HuggingFace: `HUGGINGFACEHUB_API_TOKEN`

## Output

### Experiment Output
Results are saved as CSV files with detailed information about:
- Original questions and correct answers
- Baseline model responses
- First opinions given to the model
- Second opinions provided by the model
- Contextual factors used in the prompts
- Accuracy metrics

### Evaluation Output
The evaluation pipeline generates:
- Latex table of Performance (Table 1 in the paper)
- GEE model with coefficient plot (Table 1 in the paper) and significance latex table
- LaTeX table of AUC and Brier score metrics with confidence intervals (Table 2 in the paper)
- Visualization plots for:
  - Feature ablation studies
  - Computational order analysis (Figure 4 in the paper)
  - Cross-model comparison

## Data Availability

Due to size constraints, the raw experiment data is not included in this repository. However, processed analysis results for the NEJM dataset are available in the `processed` directory.
