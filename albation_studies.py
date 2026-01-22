import torch
import random
import csv
import time
import math
import statistics
import os
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm


# Option 1: Use environment variables (recommended)



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RANDOM_SEED = 42
SEQ_LENGTH = 510
NUM_SEQUENCES = 500
TEMPERATURE = 1.0
TOP_K = 0
TOP_P = 1.0

STEP_VALUES = [10, 20, 40,80]
MASK_RATIOS = [0.10, 0.15, 0.20]

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

BASE_OUTDIR = "ablations"
os.makedirs(BASE_OUTDIR, exist_ok=True)

# ============================================================
# MODEL CONFIG
# ============================================================

MODELS = {
    "fnet": {
        "checkpoint": "Rogue05/run_023_lr7e-05_wd0.05_bs2_ga4_len512",
        "tokenizer": "facebook/esm2_t6_8M_UR50D"
    },
    "esm": {
        "checkpoint": "PES1UG23AM235/run_021_lr7e-05_wd0.05_bs1_ga4_len512",
        "tokenizer": "facebook/esm2_t6_8M_UR50D"
    }
}

# ============================================================
# UTILS
# ============================================================

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)

def random_init_sequence(length):
    return "".join(random.choices(AMINO_ACIDS, k=length))

def apply_mask(input_ids, tokenizer, mask_ratio):
    """Apply masking to a sequence, avoiding special tokens"""
    valid_positions = list(range(1, len(input_ids) - 1))
    num_mask = max(1, int(len(valid_positions) * mask_ratio))
    mask_positions = random.sample(valid_positions, num_mask)
    for pos in mask_positions:
        input_ids[pos] = tokenizer.mask_token_id
    return mask_positions

def calculate_posthoc_perplexity(model, tokenizer, sequence):
    """Calculate perplexity of a complete sequence"""
    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True).to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(out.loss).item()

# ============================================================
# GENERATION FUNCTION
# ============================================================

def generate_sequence(model, tokenizer, num_steps, mask_ratio, seq_seed=None):
    """
    Generate a sequence using iterative refinement.
    
    Args:
        model: The masked language model
        tokenizer: The tokenizer
        num_steps: Number of refinement iterations
        mask_ratio: Fraction of positions to mask per iteration
        seq_seed: Optional seed for this specific sequence (for reproducibility)
    
    Returns:
        tuple: (sequence, total_time, avg_latency, self_ppl, memory_mb)
    """
    # Set sequence-specific seed if provided
    if seq_seed is not None:
        set_seed(seq_seed)
    
    # Reset memory tracking
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # Initialize random sequence
    seq = random_init_sequence(SEQ_LENGTH)
    input_ids = tokenizer(
        seq, return_tensors="pt", add_special_tokens=True
    )["input_ids"][0].to(DEVICE)

    step_losses = []
    iter_times = []

    start_time = time.time()

    for step in range(num_steps):
        # Create a copy to mask
        masked_ids = input_ids.clone()
        mask_positions = apply_mask(masked_ids, tokenizer, mask_ratio)

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        iter_start = time.time()

        with torch.no_grad():
            outputs = model(masked_ids.unsqueeze(0))
            logits = outputs.logits[0]

            # Sample predictions for masked positions
            for pos in mask_positions:
                token_logits = logits[pos] / TEMPERATURE

                # Apply top-k filtering if enabled
                if TOP_K > 0:
                    threshold = torch.topk(token_logits, TOP_K)[0][-1]
                    token_logits[token_logits < threshold] = float("-inf")

                # Apply nucleus (top-p) filtering if enabled
                if TOP_P < 1.0:
                    sorted_logits, sorted_indices = torch.sort(token_logits, descending=True)
                    probs = torch.softmax(sorted_logits, dim=-1)
                    cum_probs = torch.cumsum(probs, dim=-1)
                    remove = cum_probs > TOP_P
                    remove[1:] = remove[:-1].clone()
                    remove[0] = False
                    token_logits[sorted_indices[remove]] = float("-inf")

                # Sample from filtered distribution
                probs = torch.softmax(token_logits, dim=-1)
                pred_id = torch.multinomial(probs, 1).item()
                
                # Track loss (negative log probability)
                pred_prob = probs[pred_id].item()
                if pred_prob > 0:
                    step_losses.append(-math.log(pred_prob))
                else:
                    step_losses.append(float('inf'))  # Handle edge case

                # Update sequence with predicted token
                token = tokenizer.decode([pred_id]).strip()
                if token in AMINO_ACIDS:
                    input_ids[pos] = pred_id

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        iter_times.append(time.time() - iter_start)

    total_time = time.time() - start_time
    avg_latency = statistics.mean(iter_times) if iter_times else 0.0
    
    # Calculate self-perplexity (geometric mean of per-token perplexities)
    finite_losses = [l for l in step_losses if math.isfinite(l)]
    if finite_losses:
        self_ppl = math.exp(statistics.mean(finite_losses))
    else:
        self_ppl = float('inf')

    # Decode final sequence
    decoded = tokenizer.decode(input_ids, skip_special_tokens=True)
    decoded = "".join([c for c in decoded if c in AMINO_ACIDS])

    # Measure peak memory
    if DEVICE == "cuda":
        memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        memory_mb = 0.0

    return decoded, total_time, avg_latency, self_ppl, memory_mb

# ============================================================
# ABLATION RUNNER
# ============================================================

def run_ablation(model_name, ablation_type):
    """
    Run ablation study for a specific model and ablation type.
    
    Args:
        model_name: "fnet" or "esm"
        ablation_type: "steps" or "mask_ratio"
    """
    print(f"\n{'='*60}")
    print(f"Running {model_name.upper()} | Ablation: {ablation_type}")
    print(f"{'='*60}")

    cfg = MODELS[model_name]
    token = HF_TOKENS[model_name]

    # Load model and tokenizer
    print(f"Loading tokenizer: {cfg['tokenizer']}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["tokenizer"], token=token
    )
    print(f"Loading model: {cfg['checkpoint']}")
    model = AutoModelForMaskedLM.from_pretrained(
        cfg["checkpoint"], token=token
    )
    model.to(DEVICE)
    model.eval()

    # Create output directory
    outdir = f"{BASE_OUTDIR}/{model_name}/{ablation_type}"
    os.makedirs(outdir, exist_ok=True)

    # Determine values to test
    values = STEP_VALUES if ablation_type == "steps" else MASK_RATIOS
    fixed_steps = 40
    fixed_mask = 0.15

    summary = []

    for val in values:
        print(f"\nTesting {ablation_type}={val}")
        csv_path = f"{outdir}/{ablation_type}_{val}.csv"
        self_ppls = []
        posthoc_ppls = []

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "seq_id", "gen_time", "avg_latency", "self_ppl",
                "posthoc_ppl", "memory_mb",
                "steps", "mask_ratio"
            ])

            # Generate sequences with consistent seeds
            for i in tqdm(range(NUM_SEQUENCES), desc=f"{ablation_type}={val}"):
                # Use deterministic seed for each sequence
                seq_seed = RANDOM_SEED + i
                
                steps = val if ablation_type == "steps" else fixed_steps
                mask = fixed_mask if ablation_type == "steps" else val

                seq, t, latency, self_ppl, mem = generate_sequence(
                    model, tokenizer, steps, mask, seq_seed=seq_seed
                )
                posthoc = calculate_posthoc_perplexity(model, tokenizer, seq)

                writer.writerow([i, t, latency, self_ppl, posthoc, mem, steps, mask])
                
                if math.isfinite(self_ppl):
                    self_ppls.append(self_ppl)
                if math.isfinite(posthoc):
                    posthoc_ppls.append(posthoc)

        # Calculate statistics
        if self_ppls:
            mean_self = statistics.mean(self_ppls)
            sd_self = statistics.stdev(self_ppls) if len(self_ppls) > 1 else 0.0
        else:
            mean_self = float('inf')
            sd_self = 0.0
        
        if posthoc_ppls:
            mean_posthoc = statistics.mean(posthoc_ppls)
            sd_posthoc = statistics.stdev(posthoc_ppls) if len(posthoc_ppls) > 1 else 0.0
        else:
            mean_posthoc = float('inf')
            sd_posthoc = 0.0

        summary.append((val, mean_self, sd_self, mean_posthoc, sd_posthoc))
        print(f"  Self-PPL: {mean_self:.2f} ± {sd_self:.2f}")
        print(f"  Post-hoc PPL: {mean_posthoc:.2f} ± {sd_posthoc:.2f}")

    # Save summary
    with open(f"{outdir}/summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["value", "self_ppl_mean", "self_ppl_sd", 
                        "posthoc_ppl_mean", "posthoc_ppl_sd"])
        for row in summary:
            writer.writerow(row)

    print(f"\nResults saved to {outdir}/")

# ============================================================
# PLOTTING
# ============================================================

def plot_ablation(ablation_type, metric="self_ppl"):
    """
    Create comparison plots for ablation studies.
    
    Args:
        ablation_type: "steps" or "mask_ratio"
        metric: "self_ppl" or "posthoc_ppl"
    """
    plt.figure(figsize=(8, 5))

    for model_name in MODELS:
        path = f"{BASE_OUTDIR}/{model_name}/{ablation_type}/summary.csv"
        
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping {model_name}")
            continue
            
        xs, ys, yerr = [], [], []

        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                xs.append(float(row["value"]))
                ys.append(float(row[f"{metric}_mean"]))
                yerr.append(float(row[f"{metric}_sd"]))

        plt.errorbar(xs, ys, yerr=yerr, marker="o", markersize=8, 
                    capsize=5, capthick=2, label=model_name.upper(), linewidth=2)

    xlabel = "Number of Refinement Steps" if ablation_type == "steps" else "Mask Ratio"
    ylabel = "Self-Perplexity" if metric == "self_ppl" else "Post-hoc Perplexity"
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f"{ylabel} vs {xlabel}", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()

    plot_dir = f"{BASE_OUTDIR}/plots"
    os.makedirs(plot_dir, exist_ok=True)
    filename = f"{plot_dir}/{ablation_type}_{metric}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {filename}")
    plt.close()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Set global seed at start
    set_seed(RANDOM_SEED)
    
    print(f"Device: {DEVICE}")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Sequence Length: {SEQ_LENGTH}")
    print(f"Number of Sequences: {NUM_SEQUENCES}")
    print(f"Step Values: {STEP_VALUES}")
    print(f"Mask Ratios: {MASK_RATIOS}")
    
    # Run ablation studies
    for model_name in MODELS:
        run_ablation(model_name, "steps")
        run_ablation(model_name, "mask_ratio")

    # Create plots
    print("\n" + "="*60)
    print("Creating plots...")
    print("="*60)
    
    plot_ablation("steps", "self_ppl")
    plot_ablation("steps", "posthoc_ppl")
    plot_ablation("mask_ratio", "self_ppl")
    plot_ablation("mask_ratio", "posthoc_ppl")

    print("\n" + "="*60)
    print("All ablations and plots completed successfully!")
    print(f"Results saved in: {BASE_OUTDIR}/")
    print("="*60)