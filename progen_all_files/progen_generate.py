#!/usr/bin/env python3
import torch, math, csv, os, time, random, statistics
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer
from progen.progen2.models.progen.modeling_progen import ProGenForCausalLM


MODEL_ID = "pedriGavi/run_004_lr3e-05_wd0.01_bs2_ga8_len512"
HF_TOKEN = ""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42
NUM_SEQUENCES = 5
PROMPT_LENGTH = 0  
TARGET_AA_LENGTH = 510  
TEMPERATURE = 1
TOP_P = 1
TOP_K = 0
BATCH_SIZE = 1
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
FASTA_FILE = f"progen_gen_{timestamp}.fasta"
LOG_FILE   = f"progen_gen_log_{timestamp}.csv"


print(" Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = ProGenForCausalLM.from_pretrained(MODEL_ID, token=HF_TOKEN, ignore_mismatched_sizes=True)
model.to(DEVICE).eval()
print(f" Model loaded from {MODEL_ID} on {DEVICE}")

from types import MethodType
import torch.nn.functional as F

def safe_forward(self, *args, **kwargs):
    out = self.__class__.forward(self, *args, **kwargs)
    if hasattr(out, "logits"):
        logits = out.logits
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        logits = torch.clamp(logits, min=-50, max=50)
        out.logits = logits
    return out

model.forward = MethodType(safe_forward, model)
print("Patched forward() for safe generation (logit clamp [-50, 50])")


def random_init_sequence(length):
    """For empty prompt case, returns empty string"""
    if length == 0:
        return ""
    seq = "".join(random.choices(AMINO_ACIDS, k=length))
    return " ".join(list(seq))  

@torch.no_grad()
def calc_perplexity(model, tokenizer, seq):
    spaced = " ".join(list(seq))
    inputs = tokenizer(spaced, return_tensors="pt").to(DEVICE)
    out = model(**inputs, labels=inputs["input_ids"])
    return math.exp(out.loss.item())

def compute_self_perplexity(logprobs):
    if not logprobs:
        return float('nan')
    avg_nll = -statistics.mean(logprobs)
    return math.exp(avg_nll)

def compute_summary(values):
    if not values:
        return 0, 0, (0, 0)
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0
    ci = 1.96 * (sd / math.sqrt(len(values))) if len(values) > 1 else 0
    return mean, sd, (mean - ci, mean + ci)

@torch.no_grad()
def generate_sequence_with_metrics(model, tokenizer, prompt_length, target_aa_length, temperature, top_p, top_k):
    """Generate sequence using Hugging Face generate(), while collecting metrics."""

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    prompt = random_init_sequence(prompt_length)
    if prompt_length == 0:
        if getattr(tokenizer, "bos_token_id", None) is not None:
            input_ids = torch.tensor([[tokenizer.bos_token_id]], device=DEVICE)
        else:
            first_aa = random.choice(AMINO_ACIDS)
            input_ids = tokenizer(f" {first_aa}", return_tensors="pt").input_ids.to(DEVICE)
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    gen_out = model.generate(
        input_ids=input_ids,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_new_tokens=target_aa_length, 
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
    )

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    gen_time = time.time() - t0

    # decode and clean
    decoded = tokenizer.decode(gen_out[0], skip_special_tokens=True)
    seq = "".join([c for c in decoded.replace(" ", "") if c in AMINO_ACIDS])
    seq = seq[:target_aa_length]

    # metrics
    total_tokens = gen_out.shape[1] - input_ids.shape[1]
    avg_token_latency = gen_time / max(total_tokens, 1)
    throughput = len(seq) / gen_time if gen_time > 0 else 0
    memory_used = (torch.cuda.max_memory_allocated() / 1024 ** 2) if DEVICE == "cuda" else 0.0

    # self-perplexity
    logprobs = []
    spaced = " ".join(list(seq))
    input_ids_seq = tokenizer(spaced, return_tensors="pt").to(DEVICE).input_ids
    for pos in range(1, input_ids_seq.size(1)):
        logits = model(input_ids_seq[:, :pos]).logits[:, -1, :]
        probs = torch.softmax(logits / max(temperature, 1e-8), dim=-1)
        next_tok = input_ids_seq[:, pos]
        prob = probs.gather(1, next_tok.unsqueeze(-1)).clamp_min(1e-12)
        logprobs.append(math.log(prob.item()))
    self_ppl = compute_self_perplexity(logprobs)

    # post-hoc perplexity
    post_ppl = calc_perplexity(model, tokenizer, seq)

    return seq, gen_time, avg_token_latency, throughput, self_ppl, post_ppl, memory_used



print(f"\n Generating {NUM_SEQUENCES} sequences...\n")
print(f"Target sequence length: {TARGET_AA_LENGTH} amino acids")
print(f"Prompt length: {PROMPT_LENGTH} (empty prompt - generating from scratch)\n")

csv_header = [
    "seq_id", "generation_time_sec", "avg_token_latency_ms", "throughput_tokens_per_sec",
    "sequence_length", "self_perplexity", "posthoc_perplexity", "memory_usage_mb",
    "seed", "temperature", "top_k", "top_p", "batch_size", "target_aa_length"
]

# Track metrics
gen_times, avg_latencies, throughputs = [], [], []
self_ppls, post_ppls, seq_lengths = [], [], []
memory_usage = []

with open(FASTA_FILE, "w") as fasta_out, open(LOG_FILE, "w", newline="") as csv_out:
    writer = csv.DictWriter(csv_out, fieldnames=csv_header)
    writer.writeheader()

    for i in tqdm(range(NUM_SEQUENCES), desc="Generating sequences"):
        seq_id = f"seq_{i+1:05d}"
        
        seq, gen_time, avg_latency, throughput, self_ppl, post_ppl, memory_used = generate_sequence_with_metrics(
            model, tokenizer, PROMPT_LENGTH, TARGET_AA_LENGTH, TEMPERATURE, TOP_P, TOP_K
        )

        fasta_out.write(f">{seq_id}\n{seq}\n")
        writer.writerow({
            "seq_id": seq_id,
            "generation_time_sec": round(gen_time, 4),
            "avg_token_latency_ms": round(avg_latency * 1000, 3),
            "throughput_tokens_per_sec": round(throughput, 2),
            "sequence_length": len(seq),
            "self_perplexity": round(self_ppl, 4),
            "posthoc_perplexity": round(post_ppl, 4),
            "memory_usage_mb": round(memory_used, 2),
            "seed": RANDOM_SEED,
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
            "top_p": TOP_P,
            "batch_size": BATCH_SIZE,
            "target_aa_length": TARGET_AA_LENGTH
        })

        # Store for summary
        gen_times.append(gen_time)
        avg_latencies.append(avg_latency * 1000)
        throughputs.append(throughput)
        self_ppls.append(self_ppl)
        post_ppls.append(post_ppl)
        seq_lengths.append(len(seq))
        memory_usage.append(memory_used)

        fasta_out.flush()
        csv_out.flush()

print(f"\n Generation complete!")
print(f" FASTA → {FASTA_FILE}")
print(f" Log   → {LOG_FILE}")


metrics = {
    "Generation Time (s)": gen_times,
    "Avg Token Latency (ms)": avg_latencies,
    "Throughput (tokens/s)": throughputs,
    "Sequence Length": seq_lengths,
    "Self-Perplexity": self_ppls,
    "Posthoc-Perplexity": post_ppls,
    "Memory Usage (MB)": memory_usage
}

print("\n Summary Statistics Across Sequences:")
print(f"{'Metric':<30}{'Mean':>12}{'SD':>12}{'95% CI (Lower, Upper)':>35}")
print("-" * 90)

for name, vals in metrics.items():
    mean, sd, ci = compute_summary(vals)
    print(f"{name:<30}{mean:>12.4f}{sd:>12.4f}{str((round(ci[0],4), round(ci[1],4))):>35}")

print("-" * 90)
print("\n Full statistical summary computed successfully!")
