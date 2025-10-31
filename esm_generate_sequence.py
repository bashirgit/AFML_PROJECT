import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import random
import csv
from datetime import datetime
from tqdm import tqdm
import time
import statistics

# ===============================
# CONFIGURATION
# ===============================

MODEL_ID = "PES1UG23AM235/run_021_lr7e-05_wd0.05_bs1_ga4_len512"
HF_TOKEN = ""

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
FASTA_FILE = f"generated_sequences_esm_{TIMESTAMP}.fasta"
LOG_FILE = f"generation_log_esm_{TIMESTAMP}.csv"

# ===============================
# LOAD MODEL & TOKENIZER
# ===============================

print("\n📍 Loading tokenizer and model...")

TOKENIZER_BACKBONE = "facebook/esm2_t6_8M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_BACKBONE)

model = AutoModelForMaskedLM.from_pretrained(MODEL_ID, token=HF_TOKEN)
model.to(DEVICE)
model.eval()

print(f"✅ Model loaded from: {MODEL_ID}")
print(f"✅ Running on device: {DEVICE}\n")

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# ===============================
# FUNCTIONS
# ===============================

def random_init_sequence(length):
    """Generate a random amino acid sequence of given length."""
    return "".join(random.choices(AMINO_ACIDS, k=length))


def calculate_perplexity(model, tokenizer, sequence):
    """Compute perplexity for a given protein sequence."""
    input_ids = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)["input_ids"].to(DEVICE)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return torch.exp(loss).item()


def generate_sequence(model, tokenizer, seq_length, num_iterations,
                      temperature, top_k, top_p):
    """Generate a protein sequence via iterative masked LM sampling."""
    
    # Track memory BEFORE generation starts
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)
        initial_memory = torch.cuda.memory_allocated(DEVICE) / 1024**2  # MB

    # Initialize random sequence
    seq = list(random_init_sequence(seq_length))
    input_ids = tokenizer("".join(seq), return_tensors="pt", add_special_tokens=True)["input_ids"].to(DEVICE)
    input_ids = input_ids[0]
    
    # Store actual token count (excluding special tokens for fair comparison)
    actual_seq_length = len(input_ids) - 2  # Exclude CLS and SEP tokens

    iteration_latencies = []
    
    # START TIMING: Only measure generation, not perplexity
    start_time = time.time()

    for _ in range(num_iterations):
        mask_pos = random.randint(1, len(input_ids) - 2)
        input_ids[mask_pos] = tokenizer.mask_token_id

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        iter_start = time.time()
        
        with torch.no_grad():
            outputs = model(input_ids.unsqueeze(0))
            logits = outputs.logits[0, mask_pos]
            logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                threshold = torch.topk(logits, top_k)[0][..., -1]
                logits[logits < threshold] = float('-inf')

            # Top-p (nucleus) filtering
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

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        iteration_latencies.append(time.time() - iter_start)

        pred_token = tokenizer.decode([pred_id]).strip()
        if pred_token in AMINO_ACIDS:
            input_ids[mask_pos] = pred_id

    # END TIMING: Generation complete
    total_time = time.time() - start_time

    # Decode sequence (perplexity calculated separately in main loop)
    decoded_seq = tokenizer.decode(input_ids, skip_special_tokens=True)
    decoded_seq = "".join([c for c in decoded_seq if c in AMINO_ACIDS])

    # Memory usage
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated(DEVICE) / 1024**2
        memory_used = peak_memory - initial_memory
    else:
        memory_used = 0.0

    # Calculate metrics
    avg_iter_latency = statistics.mean(iteration_latencies)
    # Throughput: tokens generated per second (excluding special tokens)
    throughput = actual_seq_length / total_time if total_time > 0 else 0

    return decoded_seq, total_time, memory_used, avg_iter_latency, throughput, actual_seq_length


# ===============================
# MAIN GENERATION LOOP
# ===============================

print(f"🚀 Generating {NUM_SEQUENCES} sequences ({SEQ_LENGTH} AA, {NUM_ITERATIONS} iterations each)...\n")

csv_header = [
    'seq_id', 'sequence_length', 'generation_time_sec', 'avg_iter_latency_ms',
    'throughput_tokens_per_sec', 'memory_usage_mb', 'perplexity', 'seed',
    'temperature', 'top_k', 'top_p', 'num_iterations'
]

with open(FASTA_FILE, 'w') as fasta_out, open(LOG_FILE, 'w', newline='') as csv_out:
    csv_writer = csv.DictWriter(csv_out, fieldnames=csv_header)
    csv_writer.writeheader()

    for i in tqdm(range(NUM_SEQUENCES), desc="Generating sequences"):
        seq_id = f"seq_{i+1:05d}"

        # Generate sequence (timed separately)
        sequence, gen_time, memory_used, avg_iter_latency, throughput, actual_len = generate_sequence(
            model, tokenizer, SEQ_LENGTH, NUM_ITERATIONS,
            TEMPERATURE, TOP_K, TOP_P
        )

        # Calculate perplexity AFTER generation (not included in timing)
        perplexity = calculate_perplexity(model, tokenizer, sequence)

        fasta_out.write(f">{seq_id}\n{sequence}\n")

        csv_writer.writerow({
            'seq_id': seq_id,
            'sequence_length': actual_len,
            'generation_time_sec': round(gen_time, 4),
            'avg_iter_latency_ms': round(avg_iter_latency * 1000, 3),  # Convert to ms
            'throughput_tokens_per_sec': round(throughput, 2),
            'memory_usage_mb': round(memory_used, 2),
            'perplexity': round(perplexity, 4),
            'seed': RANDOM_SEED,
            'temperature': TEMPERATURE,
            'top_k': TOP_K,
            'top_p': TOP_P,
            'num_iterations': NUM_ITERATIONS
        })

print("\n✅ Generation complete!")
print(f"🧬 FASTA saved to: {FASTA_FILE}")
print(f"📊 Log saved to:   {LOG_FILE}")
