import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import random
import csv

from PhotonicAttention import noisify_attentions



######################### CONFIGS START #################################
model_name = "meta-llama/Llama-3.1-8B"
max_samples = None

NUM_RANDOM_SEEDS = 10
SEEDS = random.sample(range(1, 1_000_000_000), NUM_RANDOM_SEEDS)
print(f"selected random SEEDS:{SEEDS}")

noise_std_values = np.round(np.arange(0.0, 0.5, 0.01), 2)
print(f"noise std values: {noise_std_values}")

# We select the following benchmarks from MMLU
filter_subjects = [
    "college_medicine",
    "formal_logic",
    "international_law",
    "high_school_statistics",
    "college_computer_science",
]

csv_filename = "mmlu_results.csv"
fieldnames = ["seed", "noise_std", "subject", "accuracy", "num_samples"]

########################## CONFIGS END ##################################





####################### MODEL CONSTRUCTION START ########################
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

# For convenience we simply modify the eager attention implementation
# By default HF Transformers use SDPA or FlashAttention
# We set the config to eager here to use the eager implementation
model.config._attn_implementation = "eager"
######################## MODEL CONSTRUCTIONE END #########################





#################### DATASET LOADING AND PROCESSING START ################
mmlu_dataset = load_dataset(
    "json", 
    data_files={"train": "data/mmlu/five_shot_mmlu_test.json"}, 
    split="train"
)
mmlu_dataset = mmlu_dataset.filter(lambda ex: ex["subject"] in filter_subjects)

if max_samples and len(mmlu_dataset) > max_samples:
    mmlu_dataset = mmlu_dataset.select(range(max_samples))

accuracy_metric = evaluate.load("accuracy")

abcd_idx = [
    tokenizer("A", add_special_tokens=False).input_ids[0],
    tokenizer("B", add_special_tokens=False).input_ids[0],
    tokenizer("C", add_special_tokens=False).input_ids[0],
    tokenizer("D", add_special_tokens=False).input_ids[0],
]

##################### DATASET LOADING AND PROCESSING END #################





############################## CSV SETUP START ###########################
with open(csv_filename, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
############################## CSV SETUP START ###########################





######################### EXPERIMENTS LOOPS START ########################

for seed in SEEDS:
    # THis sets the right seed of interest
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    for noise_std in noise_std_values:
        # This function call replaces both original LlamaAttention
        # And the previous PhotonicAttention with the PhotonicAttention
        # with the current noise standard deviation of interest 
        noisify_attentions(model, noise_std=noise_std)

        predictions_by_subj = defaultdict(list)
        references_by_subj  = defaultdict(list)

        for ex in tqdm(
            mmlu_dataset, 
            desc=f"Seed={seed}, noise_std={noise_std}", 
            leave=False
        ):
            subject = ex["subject"]
            question = ex["input"]
            gold     = ex["output"].upper()
            
            # Not sure if this prompt is correct, but I double checked the reported
            # MMLU accuracies of Llama3.1-8B and this seems close.
            prompt = f"Q: {question}\n Answer:"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            last_idx = inputs["input_ids"].shape[1] - 1
            final_logits = outputs.logits[0, last_idx, :]
            logit_abcd = final_logits[abcd_idx]
            
            pred_idx = torch.argmax(logit_abcd).item()
            gold_idx = ["A","B","C","D"].index(gold)
            
            predictions_by_subj[subject].append(pred_idx)
            references_by_subj[subject].append(gold_idx)

        with open(csv_filename, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            all_preds = []
            all_refs  = []

            for subject in predictions_by_subj:
                preds_subj = predictions_by_subj[subject]
                refs_subj  = references_by_subj[subject]

                acc_subj = accuracy_metric.compute(
                    predictions=preds_subj, 
                    references=refs_subj
                )["accuracy"]

                writer.writerow({
                    "seed": seed,
                    "noise_std": noise_std,
                    "subject": subject,
                    "accuracy": acc_subj,
                    "num_samples": len(refs_subj),
                })

                all_preds.extend(preds_subj)
                all_refs.extend(refs_subj)

            acc_all = accuracy_metric.compute(
                predictions=all_preds, 
                references=all_refs
            )["accuracy"]

            writer.writerow({
                "seed": seed,
                "noise_std": noise_std,
                "subject": "ALL_FILTERED",
                "accuracy": acc_all,
                "num_samples": len(all_refs),
            })

########################## EXPERIMENTS LOOPS END #########################





print(f"Experiments finished, results are written to {csv_filename}.")
