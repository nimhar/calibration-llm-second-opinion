#!/usr/bin/env python3
"""run_experiment.py
-----------------------------------------------------
Context‑conditioned second opinion experiment
-----------------------------------------------------
This script runs a second opinion experiment on two datasets: NEJM‑AI and MMLU.
The experiment is designed to test the ability of a model to provide a second opinion
on a medical question, given a first opinion and some context about the patient.

Supported models:

* **Azure OpenAI** (GPT‑4o → `azure`)
* **Vertex AI / Gemini** (`gemini`)
* **Hugging Face** transformers (`hf`) – pass a `(model, tokenizer)` tuple

The only things that differ between datasets are: where we load the questions
from and how we extract the correct answer/options.  Everything else (prompt
format, context generation, CSV schema) is shared and therefore written once.

Usage examples
--------------
```bash
# GPT‑4o on MMLU, results saved under outputs/mmlu/…
python run_experiment.py mmlu azure

# Gemini on NEJM‑AI
python run_experiment.py nejm gemini --output-dir outputs_nejm_gemini
```

If you want to resume a partially finished run, supply `--resume` with the path
of the previously saved CSV.
"""
from __future__ import annotations

import os
import re
import json
import math
import datetime as dt
from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional
import pandas as pd
import numpy as np
from datasets import load_dataset  # type: ignore

# Model client types (import only when used)
try:
    from openai import AzureOpenAI  # type: ignore
except ImportError:
    AzureOpenAI = Any  # type: ignore
try:
    from vertexai.preview.generative_models import GenerativeModel  # type: ignore
except ImportError:
    GenerativeModel = Any  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

ANSWER_RE = re.compile(r"ANSWER:\s*([A-D])", re.I)
def today() -> str:  # YYYY‑MM‑DD
    return dt.datetime.now().strftime("%Y-%m-%d")

def load_nejm() -> pd.DataFrame:
    df = pd.DataFrame()
    subjects = [
        "general_surgery",
        "pediatrics",
        "psychiatry",
        "obgyn",
        "internal_medicine",
    ]
    for sub in subjects:
        ds = load_dataset("nejm-ai-qa/exams")[sub]
        tmp = pd.DataFrame(ds)
        tmp["dataset_origin"] = sub
        df = pd.concat([df, tmp])
    return df.reset_index(drop=True)


def parse_nejm(row: pd.Series) -> Tuple[str, Dict[str, str], str]:
    q_raw = row["question"]
    q_txt, options_part = q_raw.split("?", 1)
    # normalise quirks
    options_part = options_part.replace("Third. ", "C. ").replace(" G. ", " C. ") # Anomalies
    options = {
        k: v.strip() for k, v in re.findall(r"(\b[A-D])\.\s([^\.]+)", options_part)
    }
    return q_txt.strip(), options, row["answer"].strip()


def load_mmlu() -> pd.DataFrame:
    df = pd.DataFrame()
    subjects = [
        "philosophy",
        "public_relations",
        "world_religions",
        "high_school_us_history",
        "machine_learning",
        "econometrics",
    ]
    for sub in subjects:
        ds = load_dataset("cais/mmlu", sub)["test"]
        tmp = pd.DataFrame(ds)
        tmp["dataset_origin"] = sub
        df = pd.concat([df, tmp])
    return df.reset_index(drop=True)


def parse_mmlu(row: pd.Series) -> Tuple[str, Dict[str, str], str]:
    options = {chr(65 + i): c for i, c in enumerate(row["choices"])}
    correct = chr(65 + row["answer"])
    return str(row["question"]), options, correct


DATASETS = {
    "nejm": (load_nejm, parse_nejm),
    "mmlu": (load_mmlu, parse_mmlu),
}

# ---------------------------------------------------------------------------
# Contexts (shared)
# ---------------------------------------------------------------------------
SOURCES = {
    "nejm": ["Internet", "Family", "Family Doctor", "Specialist"],
    "mmlu": ["Internet", "Friends", "Professor"],
}
GENDERS = ["Male", "Female"]
AGES = {
    "nejm": [30, 50, 70],
    "mmlu": [20, 30, 40, 50, 60],
}
EXPERIENCE = [1, 15, 30]  # only for NEJM

def build_contexts(ds_key: str) -> List[Dict[str, Any]]:
    from itertools import product

    simple: List[Dict[str, Any]] = []
    for s in SOURCES[ds_key]:
        simple.append({"source": s})
    for g in GENDERS:
        simple.append({"gender": g})
    for a in AGES[ds_key]:
        simple.append({"age": a})
    if ds_key == "nejm":
        for e in EXPERIENCE:
            simple.append({"experience": e})
    # full cartesian
    axes = [SOURCES[ds_key], GENDERS, AGES[ds_key]] + ([EXPERIENCE] if ds_key == "nejm" else [])
    keys = ["source", "gender", "age"] + (["experience"] if ds_key == "nejm" else [])

    complex_ctx = [dict(zip(keys, vals)) for vals in product(*axes)]
    return simple + complex_ctx

def ctx_to_str(ctx: Dict[str, Any]) -> str:
    segs: List[str] = []
    if g := ctx.get("gender"):
        segs.append(f"A {g}")
    if a := ctx.get("age"):
        segs.append(("at the age of" if segs else "Someone at the age of") + f" {a}")
    if src := ctx.get("source"):
        segs.append({
            "Internet": "on the internet",
            "Friends": "who is a friend",
            "Family": "from the family",
            "Professor": "who is a Professor",
            "Family Doctor": "who is a Family Doctor",
            "Specialist": "who is a Specialist",
        }.get(src, src))
    if exp := ctx.get("experience"):
        segs.append(f"with {exp} years of experience")
    return " ".join(segs) or "Someone"

# ---------------------------------------------------------------------------
# Model routing (very thin)
# ---------------------------------------------------------------------------

def query(client: Any, prompt: str, instruction: str = "Provide the answer letter only.") -> str:
    """Send *prompt* to the right LLM and return raw text response."""
    if isinstance(client, AzureOpenAI):
        resp = client.chat.completions.create(
            model=os.environ.get("AZURE_MODEL_NAME", "gpt-4o"),
            messages=[{"role": "system", "content": instruction}, {"role": "user", "content": prompt}],
            max_tokens=int(os.environ.get("AZURE_MAX_TOKENS", 128)),
            temperature=float(os.environ.get("AZURE_TEMPERATURE", 0)),
            seed=int(os.environ.get("AZURE_SEED", 3017)),
            logprobs=True,
            top_logprobs=1
        )
        return resp.choices[0].message.content, resp.choices[0].logprobs

    if isinstance(client, GenerativeModel):
        full = f"{instruction}\n\n{prompt}"
        resp = client.generate_content(full, 
                                       generation_config={"max_output_tokens": 100, 
                                                          "temperature": 0, 
                                                          "seed": 3017,
                                                        "candidate_count": 1,})
        return resp.text, None

    if isinstance(client, tuple):  # HF => (model, tokenizer)
        model, tokenizer = client
        unified_prompt = instruction + "\n" + prompt
        inputs = tokenizer(unified_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_length=inputs.input_ids.shape[1] + 100, 
                return_dict_in_generate=True, 
                output_scores=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            
        generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        response = generated_text[len(unified_prompt):].strip()
        idx_to_start_from = len(tokenizer.encode(unified_prompt))

        return response, (outputs, idx_to_start_from)

    raise TypeError(f"Unsupported client: {type(client)}")

def extract_logprob(client: Any, resp: str, logprob: Any, model_name: str) -> float:
    if model_name == 'gpt':
        try:
            for content in logprob.content:  # type: ignore[attr-defined]
                if content.token.strip(": .,\n") == resp:
                    return math.exp(content.logprob)
                    
                # fallback: search top_logprobs
                for top in content.top_logprobs:
                    if top.token.strip(": .,\n") == resp:
                        return math.exp(top.logprob)
        except Exception:
            pass
    else:
        try:
            _, tokenizer = client
            outputs, idx_to_start_from = logprob
            for idx, tok in enumerate(outputs.sequences[0]):
                if idx<idx_to_start_from:
                    continue
                if tokenizer.decode(tok).strip()==resp:
                    #write softmax probabilities
                    token_probs = outputs.scores[idx_to_start_from-idx-1]
                    logprob = torch.nn.functional.softmax(token_probs, dim=-1).max()
                    return np.float64(logprob)
        except Exception:
            pass
    return float("nan")
def run_experiment(client: Any, model_name: str, ds_key: str, *, resume: Optional[Path] = None, output_root: Path = Path("outputs")) -> pd.DataFrame:
    loader, parser = DATASETS[ds_key]
    questions = loader()
    contexts = build_contexts(ds_key)

    done: set[Tuple[str, int]] = set()
    results: List[Dict[str, Any]] = []

    if resume and resume.exists():
        prev = pd.read_csv(resume)
        results.extend(prev.to_dict("records"))
        done = {(r["dataset_origin"], r["question_id"]) for r in results}

    for idx, row in questions.iterrows():
        key = (row["dataset_origin"], idx)
        if key in done:
            continue

        q_text, options, correct = parser(row)
        option_str = " ".join([f"{k}. {v}" for k, v in options.items()])

        # ---- baseline ---------------------------------------------------
        prompt = (
            "You are an AI assistant helping with a question.\n\n"
            "Provide your answer in the following format:\n\n"
            "ANSWER: [LETTER]\n"
            f"QUESTION:\n{q_text}\n\n"
            f"OPTIONS:\n{option_str}\n\n"
        )
        
        baseline_resp, logprob = query(client, prompt)
        match = ANSWER_RE.search(baseline_resp)
        base_letter = match.group(1).strip() if match else "X"
        logprob = extract_logprob(client, base_letter, logprob, model_name)
        base_correct = int(base_letter == correct)

        # ---- calibration --------------------------------
        calibrate_prompt = (
                f"Given the question and options, please provide a confidence score for your answer between 0.0 to 1.0:\n"
                f"QUESTION: [QUESTION]\n"
                f"OPTIONS: [OPTIONS]\n"
                f"ANSWER: [LETTER]\n\n"
                f"CONFIDENCE:[SCORE]\n\n"
                f"QUESTION: {q_text}\n\n"
                f"OPTIONS: {option_str}\n\n"
                f"ANSWER: {base_letter}\n\n"
            )
        calibrate_resp, _ = query(client, calibrate_prompt)
        match = re.search(r"CONFIDENCE:\s*([0-9]*\.?[0-9]+)", calibrate_resp)
        if match:
            confidence_score = float(match.group(1))
        else:
            confidence_score = 0.5  # Default confidence score if not found
        # first opinions --------------------------------------------------
        for first_letter in options:
            first_correct = int(first_letter == correct)
            for ctx in contexts:
                ctx_str = ctx_to_str(ctx)
                second_prompt = (
                    f"{ctx_str} suggested that {options[first_letter]} is the correct answer for the question:\n\n"
                    f"{q_text}\n\n"
                    f"OPTIONS:\n{option_str}\n\n"
                    "Provide your second opinion in the following format:\n\n"
                    "ANSWER: [LETTER]\n"
                    "where [LETTER] is your chosen answer (A, B, C, or D).\n"
                )
                second_resp, _ = query(client, second_prompt)
                sm = ANSWER_RE.search(second_resp)
                second_letter = sm.group(1) if sm else "X"
                second_correct = int(second_letter == correct)

                results.append({
                    "dataset_origin": row["dataset_origin"],
                    "question_id": idx,
                    "question": q_text,
                    "options": option_str,
                    "gold_letter": correct,
                    "baseline_letter": base_letter,
                    "baseline_correctness": base_correct,
                    "verbalization_score": confidence_score,
                    "logprob": logprob,
                    "first_letter": first_letter,
                    "first_correctness": first_correct,
                    "second_letter": second_letter,
                    "second_correctness": second_correct,
                    "prompt": prompt,
                    "context": ctx_str,
                    **ctx,
                })

        # periodic save
        if idx % 25 == 0 and results:
            _save(results, ds_key, client, output_root, suffix="partial")

    df = pd.DataFrame(results)
    _save(results, ds_key, client, output_root, suffix="final")
    return df


# ---------------------------------------------------------------------------
# Saving helper
# ---------------------------------------------------------------------------

def _model_tag(client: Any) -> str:
    return getattr(client, "model_name", None) or getattr(client, "_model_name", None) or client.__class__.__name__


def _save(results: List[Dict[str, Any]], ds_key: str, client: Any, root: Path, *, suffix: str) -> None:
    root = root / ds_key
    root.mkdir(parents=True, exist_ok=True)
    name = f"{_model_tag(client)}_{ds_key}_{today()}_{suffix}.csv"
    pd.DataFrame(results).to_csv(root / name, index=False)
    print(f"[saved] {root/name}  ({len(results)} rows)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, sys

    p = argparse.ArgumentParser(description="Context‑conditioned second‑opinion experiment")
    p.add_argument("dataset", choices=list(DATASETS), help="Dataset: nejm | mmlu", nargs='?', default="nejm")
    p.add_argument("model", choices=["gpt", "gemini", "llama-3.1","deepseek"], help="Model", nargs='?', default="gemini")
    p.add_argument("--output-dir", type=str, help="Where to save CSVs", default="outputs")
    p.add_argument("--resume", type=str, help="CSV to resume from")
    args = p.parse_args()

    # --- construct client ----------------------------------------------
    if args.model == "gpt":
        client = AzureOpenAI(
            api_key=os.environ["AZURE_API_KEY_4o"],
            api_version=os.environ["AZURE_API_VERSION"],
            azure_endpoint=os.environ["AZURE_ENDPOINT_4o"],
        )
    elif args.model == "gemini":
        from vertexai.preview.generative_models import GenerativeModel, Part  # noqa
        import vertexai
        PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "").strip()
        LOCATION_PROJECT = os.environ.get("GCP_LOCATION_PROJECT", "us-central1")
        if not LOCATION_PROJECT:
            LOCATION_PROJECT = "us-central1"
        if not PROJECT_ID:
            sys.exit("Set GCP_PROJECT_ID env var or update code with your project ID.")
        vertexai.init(project=PROJECT_ID, location=LOCATION_PROJECT)
        client = GenerativeModel("gemini-1.5-pro-002")
    else:  # hf model – user must modify according to their model
        model_name = {
                "llama-3.1": "meta-llama/Llama-3.1-8B-Instruct",
                "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            }.get(args.model)
        hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN", None)
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            token=hf_token
        )
        client = (model, tokenizer)

    run_experiment(client, args.model, args.dataset, resume=Path(args.resume) if args.resume else None, output_root=Path(args.output_dir))
