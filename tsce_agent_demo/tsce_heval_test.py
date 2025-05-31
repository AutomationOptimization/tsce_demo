#!/usr/bin/env python3
"""
evaluate_halueval_tsce_azure.py

Run TSCE and baseline classification on the HaluEval hallucination evaluation
benchmark using Azure OpenAI services exclusively.
"""
import os
import json
import random
import argparse

# Optional progress bar
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x

import openai
from tsce_chat import TSCEChat

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate HaluEval with TSCE and baseline (Azure OpenAI only)"
    )
    parser.add_argument(
        "--task",
        choices=["general", "qa", "dialogue", "summarization", "all"],
        default="all",
        help="Which task split to evaluate",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "tsce", "both"],
        default="both",
        help="Whether to run baseline only, TSCE only, or both",
    )
    parser.add_argument(
        "--azure-endpoint",
        default=os.getenv("AZURE_OPENAI_ENDPOINT"),
        help="Azure OpenAI endpoint (e.g. https://your-resource.openai.azure.com)",
    )
    parser.add_argument(
        "--azure-key",
        default=os.getenv("AZURE_OPENAI_KEY"),
        help="Azure OpenAI API key",
    )
    parser.add_argument(
        "--azure-api-version",
        default=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        help="Azure OpenAI API version",
    )
    parser.add_argument(
        "--azure-deployment",
        required=True,
        help="Azure OpenAI chat deployment name (used for both baseline & TSCE)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (only used in non‐‘general’ splits)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to save per‐split and aggregate JSON outputs",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="If set, only evaluate this many examples per split (for quick tests)",
    )
    args = parser.parse_args()

    # ensure required Azure settings
    if not args.azure_endpoint or not args.azure_key:
        parser.error("You must supply --azure-endpoint and --azure-key (or set AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_KEY).")
    return args

def load_data(task: str) -> list:
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    file_map = {
        "general":      "general_data.json",
        "qa":           "qa_data.json",
        "dialogue":     "dialogue_data.json",
        "summarization":"summarization_data.json",
    }
    path = os.path.join(data_dir, file_map[task])
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def build_prompt(sample: dict, task: str) -> str:
    if task == "general":
        return (
            f"User query: {sample['user_query']}\n\n"
            f"Response: {sample['chatgpt_response']}\n\n"
            "Does the above response contain hallucinations? Answer \"Yes\" or \"No\"."
        )
    if task == "qa":
        return (
            f"Knowledge: {sample['knowledge']}\n\n"
            f"Question: {sample['question']}\n\n"
            f"Answer: {sample['text']}\n\n"
            "Does the above answer contain hallucinations? Answer \"Yes\" or \"No\"."
        )
    if task == "dialogue":
        history = "\n".join(sample["dialogue_history"])
        return (
            f"Knowledge: {sample['knowledge']}\n\n"
            "Dialogue History:\n"
            f"{history}\n\n"
            f"Response: {sample['text']}\n\n"
            "Is the response hallucinated? Answer \"Yes\" or \"No\"."
        )
    if task == "summarization":
        return (
            f"Document:\n{sample['document']}\n\n"
            f"Summary: {sample['text']}\n\n"
            "Does the summary contain hallucinated content? Answer \"Yes\" or \"No\"."
        )
    raise ValueError(f"Unsupported task: {task}")

def normalize_pred(pred: str) -> str:
    p = pred.strip().lower()
    if p.startswith("yes"):
        return "Yes"
    if p.startswith("no"):
        return "No"
    return "Unknown"

def evaluate_task(
    task: str,
    samples: list,
    mode: str,
    tsce: TSCEChat | None,
    azure_client: openai.AzureOpenAI,
    deployment: str,
) -> list:
    results = []
    for sample in tqdm(samples, desc=f"Evaluating {task}"):
        # ground truth
        if task == "general":
            label = "Yes" if sample.get("hallucination", False) else "No"
        else:
            # for non‐general splits, HaluEval provides right vs hallucinated; we randomly choose one
            if random.random() < 0.5:
                key = next(k for k in ("right_answer","right_response","right_summary") if k in sample)
                sample["text"] = sample[key]
                label = "No"
            else:
                key = next(k for k in ("hallucinated_answer","hallucinated_response","hallucinated_summary") if k in sample)
                sample["text"] = sample[key]
                label = "Yes"

        prompt = build_prompt(sample, task)

        baseline_pred = None
        tsce_pred      = None

        if mode in ("baseline", "both"):
            resp = azure_client.chat.completions.create(
                model=deployment,
                messages=[{"role":"user","content":prompt}],
                temperature=0.0,
            ).model_dump()
            raw = resp["choices"][0]["message"]["content"]
            print(f"\n[DEBUG general] prompt=\n{prompt}\n→ raw baseline: {raw!r}")
            baseline_pred = normalize_pred(raw)
            baseline_pred = normalize_pred(resp["choices"][0]["message"]["content"])

        if mode in ("tsce", "both"):
            reply = tsce(prompt)
            print(f"[DEBUG general] raw TSCE: {reply.content!r}")
            tsce_pred = normalize_pred(reply.content)

        results.append({
            "task": task,
            "label": label,
            "baseline_pred": baseline_pred,
            "tsce_pred": tsce_pred,
        })

    # print accuracies
    if mode in ("baseline","both"):
        corr = sum(r["baseline_pred"] == r["label"] for r in results)
        print(f"Baseline accuracy ({task}): {corr}/{len(results)} = {corr/len(results):.4f}")
    if mode in ("tsce","both"):
        corr = sum(r["tsce_pred"] == r["label"] for r in results)
        print(f"TSCE accuracy    ({task}): {corr}/{len(results)} = {corr/len(results):.4f}")

    return results

def main():
    args = parse_args()
    random.seed(args.seed)

    # build Azure client
    azure_client = openai.AzureOpenAI(
        api_key        = args.azure_key,
        azure_endpoint = args.azure_endpoint,
        api_version    = args.azure_api_version,
    )

    # TSCEChat with explicit Azure client & deployment
    tsce = None
    if args.mode in ("tsce","both"):
        tsce = (
            TSCEChat(
                client        = azure_client,
                deployment_id = args.azure_deployment
            )
            if args.mode in ("tsce","both")
            else None
        )

    tasks = (
        [args.task]
        if args.task != "all"
        else ["general", "qa", "dialogue", "summarization"]
    )

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []

    for task in tasks:
        samples = load_data(task)
        if args.max_samples is not None:
            samples = samples[: args.max_samples]
        res     = evaluate_task(
            task, samples, args.mode, tsce,
            azure_client, args.azure_deployment
        )
        out_path = os.path.join(
            args.output_dir,
            f"{task}_{args.mode}_{args.azure_deployment}_results.json"
        )
        with open(out_path, "w", encoding="utf-8") as outf:
            json.dump(res, outf, indent=2)
        all_results.extend(res)

    # save aggregate
    agg_path = os.path.join(args.output_dir, "all_results.json")
    with open(agg_path, "w", encoding="utf-8") as outf:
        json.dump(all_results, outf, indent=2)

    print(f"\nSaved {len(all_results)} judgments → {args.output_dir}")

if __name__ == "__main__":
    main()
