import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import random
import csv
from datetime import datetime
from tqdm import tqdm
import time
import math
import statistics
import os


MODEL_ID = "Rogue05/run_023_lr7e-05_wd0.05_bs2_ga4_len512"
HF_TOKEN = ""  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

MAX_MODEL_LENGTH = 512
SEQ_LENGTH = 510
NUM_SEQUENCES = 5000
NUM_ITERATIONS = 40

RANDOM_SEED = 42
TEMPERATURE = 1
TOP_K = 0
TOP_P = 1
BATCH_SIZE = 1

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(RANDOM_SEED)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
FASTA_FILE = f"fnet_generated_sequences_{TIMESTAMP}.fasta"
LOG_FILE = f"fnet_generation_log_{TIMESTAMP}.csv"


print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
model = AutoModelForMaskedLM.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN)
model.to(DEVICE)
model.eval()
print(f"Model loaded on {DEVICE}")

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def random_init_sequence(length):
    return "".join(random.choices(AMINO_ACIDS, k=length))

def calculate_perplexity(model, tokenizer, sequence):
    """Post-hoc perplexity on final generated sequence"""
    input_ids = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)["input_ids"].to(DEVICE)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
    return perplexity

def generate_sequence(model, tokenizer, seq_length=120, num_iterations=40,
                      temperature=1.0, top_k=0, top_p=1.0):
    
    # Reset memory stats at the beginning
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    start_time = time.time()

    seq = list(random_init_sequence(seq_length))
    input_ids = tokenizer("".join(seq), return_tensors="pt", add_special_tokens=True)["input_ids"].to(DEVICE)
    input_ids = input_ids[0]

    per_token_latencies = []
    self_logprobs = []

    for _ in range(num_iterations):
        mask_pos = random.randint(1, len(input_ids) - 2)
        input_ids[mask_pos] = tokenizer.mask_token_id

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        iter_start = time.time()

        with torch.no_grad():
            outputs = model(input_ids.unsqueeze(0))
            logits = outputs.logits[0, mask_pos]

            logits = logits / temperature
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = float('-inf')

            probs = torch.softmax(logits, dim=-1)
            pred_id = torch.multinomial(probs, 1).item()
            log_prob = torch.log(probs[pred_id] + 1e-9).item()
            self_logprobs.append(log_prob)

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        iter_end = time.time()
        per_token_latencies.append(iter_end - iter_start)

        pred_token = tokenizer.decode([pred_id]).strip()
        if pred_token in AMINO_ACIDS:
            input_ids[mask_pos] = pred_id

    decoded_seq = tokenizer.decode(input_ids, skip_special_tokens=True)
    decoded_seq = "".join([c for c in decoded_seq if c in AMINO_ACIDS])

    total_time = time.time() - start_time
    avg_token_latency = sum(per_token_latencies) / len(per_token_latencies)
    throughput = len(input_ids) / total_time

    if len(self_logprobs) > 0:
        self_perplexity = math.exp(-sum(self_logprobs) / len(self_logprobs))
    else:
        self_perplexity = float('nan')

    if DEVICE == "cuda":
        memory_used = torch.cuda.max_memory_allocated() / (1024 ** 2)  
    else:
        memory_used = 0.0  

    return decoded_seq, total_time, avg_token_latency, throughput, self_perplexity, memory_used


def compute_summary(values):
    """Return mean, SD, and 95% CI (mean ± 1.96*SEM)."""
    n = len(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if n > 1 else 0
    ci_margin = 1.96 * (sd / math.sqrt(n)) if n > 1 else 0
    return mean, sd, (mean - ci_margin, mean + ci_margin)


# Warm-up runs
print("Performing warm-up runs...")
_ = generate_sequence(
    model, tokenizer, SEQ_LENGTH, NUM_ITERATIONS, TEMPERATURE, TOP_K, TOP_P
)
_ = generate_sequence(
    model, tokenizer, SEQ_LENGTH, NUM_ITERATIONS, TEMPERATURE, TOP_K, TOP_P
)
_ = generate_sequence(
    model, tokenizer, SEQ_LENGTH, NUM_ITERATIONS, TEMPERATURE, TOP_K, TOP_P
)


print(f"\nGenerating {NUM_SEQUENCES} sequences...")
print(f"Sequence length: {SEQ_LENGTH}")
print(f"Iterations per sequence: {NUM_ITERATIONS}")
print(f"Output FASTA: {FASTA_FILE}")
print(f"Output Log: {LOG_FILE}\n")

csv_header = [
    'seq_id', 'generation_time_sec', 'avg_token_latency_ms', 'throughput_tokens_per_sec',
    'sequence_length', 'self_perplexity', 'posthoc_perplexity', 'memory_usage_mb',
    'seed', 'temperature', 'top_k', 'top_p', 'batch_size', 'num_iterations'
]

# Store metrics for summary
gen_times, avg_latencies, throughputs = [], [], []
self_ppls, posthoc_ppls, seq_lengths = [], [], []
memory_usage = []

with open(FASTA_FILE, 'w') as fasta_out, open(LOG_FILE, 'w', newline='') as csv_out:
    csv_writer = csv.DictWriter(csv_out, fieldnames=csv_header)
    csv_writer.writeheader()

    for i in tqdm(range(NUM_SEQUENCES), desc="Generating sequences"):
        seq_id = f"seq_{i+1:05d}"

        seq, total_time, avg_latency, throughput, self_ppl, memory_used = generate_sequence(
            model, tokenizer, SEQ_LENGTH, NUM_ITERATIONS, TEMPERATURE, TOP_K, TOP_P
        )
        posthoc_ppl = calculate_perplexity(model, tokenizer, seq)

        fasta_out.write(f">{seq_id}\n{seq}\n")
        csv_writer.writerow({
            'seq_id': seq_id,
            'generation_time_sec': round(total_time, 4),
            'avg_token_latency_ms': round(avg_latency * 1000, 3),
            'throughput_tokens_per_sec': round(throughput, 2),
            'sequence_length': len(seq),
            'self_perplexity': round(self_ppl, 4),
            'posthoc_perplexity': round(posthoc_ppl, 4),
            'memory_usage_mb': round(memory_used, 2),
            'seed': RANDOM_SEED,
            'temperature': TEMPERATURE,
            'top_k': TOP_K,
            'top_p': TOP_P,
            'batch_size': BATCH_SIZE,
            'num_iterations': NUM_ITERATIONS
        })

        # Track metrics
        gen_times.append(total_time)
        avg_latencies.append(avg_latency * 1000)
        throughputs.append(throughput)
        self_ppls.append(self_ppl)
        posthoc_ppls.append(posthoc_ppl)
        seq_lengths.append(len(seq))
        memory_usage.append(memory_used)

        if (i + 1) % 100 == 0:
            print(f"[{seq_id}] self-PPL: {self_ppl:.2f}, post-hoc PPL: {posthoc_ppl:.2e}, memory: {memory_used:.2f} MB")

        fasta_out.flush()
        csv_out.flush()

print(f"\n Generated {NUM_SEQUENCES} sequences")
print(f" FASTA file saved: {FASTA_FILE}")
print(f" Log file saved: {LOG_FILE}")



metrics = {
    "Generation Time (s)": gen_times,
    "Avg Token Latency (ms)": avg_latencies,
    "Throughput (tokens/s)": throughputs,
    "Sequence Length": seq_lengths,
    "Self-Perplexity": self_ppls,
    "Posthoc-Perplexity": posthoc_ppls,
    "Memory Usage (MB)": memory_usage
}

print("\n Summary Statistics Across Sequences:")
print(f"{'Metric':<30}{'Mean':>12}{'SD':>12}{'95% CI (Lower, Upper)':>35}")
print("-" * 90)
for name, values in metrics.items():
    mean, sd, ci = compute_summary(values)
    print(f"{name:<30}{mean:>12.4f}{sd:>12.4f}{str((round(ci[0],4), round(ci[1],4))):>35}")
print("-" * 90)

print("\n Generation complete with full statistical summary!")
