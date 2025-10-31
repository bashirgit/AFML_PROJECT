import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import random
import csv
from datetime import datetime
from tqdm import tqdm
import time

# ===============================
# CONFIG
# ===============================
MODEL_ID = "Rogue05/run_023_lr7e-05_wd0.05_bs2_ga4_len512"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_MODEL_LENGTH = 512
SEQ_LENGTH = 510
NUM_SEQUENCES = 5000
NUM_ITERATIONS = 40

RANDOM_SEED = 42
TEMPERATURE = 1.0
TOP_K = 0
TOP_P = 1.0
BATCH_SIZE = 1

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
FASTA_FILE = f"generated_sequences_{TIMESTAMP}.fasta"
LOG_FILE = f"generation_log_{TIMESTAMP}.csv"

# ===============================
# LOAD MODEL & TOKENIZER
# ===============================
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
model = AutoModelForMaskedLM.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN)
model.to(DEVICE)
model.eval()
print(f"Model loaded on {DEVICE}")

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# ===============================
# FUNCTIONS
# ===============================
def random_init_sequence(length):
    return "".join(random.choices(AMINO_ACIDS, k=length))

def calculate_perplexity(model, tokenizer, sequence):
    input_ids = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)["input_ids"].to(DEVICE)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
    return perplexity

def generate_sequence(model, tokenizer, seq_length=120, num_iterations=40, temperature=1.0, top_k=0, top_p=1.0):
    start_time = time.time()
    if seq_length > 510:
        raise ValueError(f"seq_length must be <= 510 (got {seq_length})")
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)
        initial_memory = torch.cuda.memory_allocated(DEVICE) / 1024**2

    seq = list(random_init_sequence(seq_length))
    input_ids = tokenizer("".join(seq), return_tensors="pt", add_special_tokens=True)["input_ids"].to(DEVICE)
    input_ids = input_ids[0]

    per_token_latencies = []  # store latency per token
    for _ in range(num_iterations):
        mask_pos = random.randint(1, len(input_ids) - 2)
        input_ids[mask_pos] = tokenizer.mask_token_id

        torch.cuda.synchronize() if torch.cuda.is_available() else None
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

        torch.cuda.synchronize() if torch.cuda.is_available() else None
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

    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated(DEVICE) / 1024**2
        memory_used = peak_memory - initial_memory
    else:
        memory_used = 0.0

    return decoded_seq, total_time, memory_used, avg_token_latency, throughput

# ===============================
# MAIN GENERATION LOOP
# ===============================
print(f"\nGenerating {NUM_SEQUENCES} sequences...")
print(f"Sequence length: {SEQ_LENGTH}")
print(f"Iterations per sequence: {NUM_ITERATIONS}")
print(f"Output FASTA: {FASTA_FILE}")
print(f"Output Log: {LOG_FILE}\n")

csv_header = [
    'seq_id', 'generation_time_sec', 'avg_token_latency_ms', 'throughput_tokens_per_sec',
    'memory_usage_mb', 'sequence_length', 'perplexity',
    'seed', 'temperature', 'top_k', 'top_p', 'batch_size', 'num_iterations'
]

with open(FASTA_FILE, 'w') as fasta_out, open(LOG_FILE, 'w', newline='') as csv_out:
    csv_writer = csv.DictWriter(csv_out, fieldnames=csv_header)
    csv_writer.writeheader()
    
    for i in tqdm(range(NUM_SEQUENCES), desc="Generating sequences"):
        seq_id = f"seq_{i+1:05d}"
        
        seq, total_time, mem_used, avg_latency, throughput = generate_sequence(
            model, tokenizer, SEQ_LENGTH, NUM_ITERATIONS, TEMPERATURE, TOP_K, TOP_P
        )

        perplexity = calculate_perplexity(model, tokenizer, seq)

        fasta_out.write(f">{seq_id}\n{seq}\n")

        csv_writer.writerow({
            'seq_id': seq_id,
            'generation_time_sec': round(total_time, 4),
            'avg_token_latency_ms': round(avg_latency * 1000, 3),
            'throughput_tokens_per_sec': round(throughput, 2),
            'memory_usage_mb': round(mem_used, 2),
            'sequence_length': len(seq),
            'perplexity': round(perplexity, 4),
            'seed': RANDOM_SEED,
            'temperature': TEMPERATURE,
            'top_k': TOP_K,
            'top_p': TOP_P,
            'batch_size': BATCH_SIZE,
            'num_iterations': NUM_ITERATIONS
        })

        fasta_out.flush()
        csv_out.flush()

print(f"\n✓ Generated {NUM_SEQUENCES} sequences")
print(f"✓ FASTA file saved: {FASTA_FILE}")
print(f"✓ Log file saved: {LOG_FILE}")
print("\nGeneration complete!")
#print("\n💡 Tip: Metrics follow best practices from:")
#print("https://apxml.com/courses/quantized-llm-deployment/chapter-3-performance-evaluation-quantized-llms/measuring-inference-latency-throughput")