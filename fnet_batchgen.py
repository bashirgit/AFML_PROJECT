import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import random
import csv
from datetime import datetime
from tqdm import tqdm
import time
import statistics

# ===============================
# CONFIG
# ===============================
MODEL_ID = "Rogue05/run_023_lr7e-05_wd0.05_bs2_ga4_len512"
HF_TOKEN = "your_token_here"  # Add your token
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_MODEL_LENGTH = 512
SEQ_LENGTH = 510
NUM_SEQUENCES = 5000
NUM_ITERATIONS = 40

RANDOM_SEED = 42
TEMPERATURE = 1.0
TOP_K = 0
TOP_P = 1.0
BATCH_SIZES_TO_TEST = [1, 4, 8, 16, 32]  # Test different batch sizes

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
FASTA_FILE = f"generated_sequences_fnet_{TIMESTAMP}.fasta"
LOG_FILE = f"generation_log_fnet_{TIMESTAMP}.csv"

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

def generate_sequence_batch(model, tokenizer, seq_length, num_iterations, temperature, top_k, top_p, batch_size=1):
    """Generate multiple protein sequences in parallel via batched iterative masked LM sampling."""
    if seq_length > 510:
        raise ValueError(f"seq_length must be <= 510 (got {seq_length})")
    
    # Initialize batch of sequences
    sequences = [list(random_init_sequence(seq_length)) for _ in range(batch_size)]
    input_ids_list = []
    
    for seq in sequences:
        ids = tokenizer("".join(seq), return_tensors="pt", add_special_tokens=True)["input_ids"].to(DEVICE)
        input_ids_list.append(ids[0])
    
    # Stack into batch
    input_ids_batch = torch.stack(input_ids_list)  # [batch_size, seq_len]
    actual_seq_length = input_ids_batch.shape[1] - 2  # Exclude CLS and SEP tokens

    iteration_latencies = []
    
    # START TIMING: Only measure generation
    start_time = time.time()
    
    for _ in range(num_iterations):
        # Randomly select mask positions for each sequence in batch
        mask_positions = [random.randint(1, input_ids_batch.shape[1] - 2) for _ in range(batch_size)]
        
        # Apply masks
        for b_idx, mask_pos in enumerate(mask_positions):
            input_ids_batch[b_idx, mask_pos] = tokenizer.mask_token_id

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        iter_start = time.time()

        with torch.no_grad():
            outputs = model(input_ids_batch)
            logits_batch = outputs.logits  # [batch_size, seq_len, vocab_size]
            
            # Process each sequence in batch
            for b_idx, mask_pos in enumerate(mask_positions):
                logits = logits_batch[b_idx, mask_pos]
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
                
                pred_token = tokenizer.decode([pred_id]).strip()
                if pred_token in AMINO_ACIDS:
                    input_ids_batch[b_idx, mask_pos] = pred_id

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        iteration_latencies.append(time.time() - iter_start)

    # END TIMING
    total_time = time.time() - start_time

    # Decode all sequences
    decoded_sequences = []
    for b_idx in range(batch_size):
        decoded_seq = tokenizer.decode(input_ids_batch[b_idx], skip_special_tokens=True)
        decoded_seq = "".join([c for c in decoded_seq if c in AMINO_ACIDS])
        decoded_sequences.append(decoded_seq)

    # Calculate metrics
    avg_iter_latency = statistics.mean(iteration_latencies)
    # Throughput: total tokens generated per second across entire batch
    total_tokens = actual_seq_length * batch_size
    throughput = total_tokens / total_time if total_time > 0 else 0
    # Per-sequence metrics
    time_per_seq = total_time / batch_size
    throughput_per_seq = actual_seq_length / time_per_seq if time_per_seq > 0 else 0

    return decoded_sequences, total_time, time_per_seq, avg_iter_latency, throughput, throughput_per_seq, actual_seq_length

# ===============================
# MAIN GENERATION LOOP
# ===============================
print(f"\nGenerating {NUM_SEQUENCES} sequences...")
print(f"Sequence length: {SEQ_LENGTH}")
print(f"Iterations per sequence: {NUM_ITERATIONS}")
print(f"Testing batch sizes: {BATCH_SIZES_TO_TEST}\n")

# Run tests for each batch size
for BATCH_SIZE in BATCH_SIZES_TO_TEST:
    print(f"\n{'='*60}")
    print(f"Testing with BATCH_SIZE = {BATCH_SIZE}")
    print(f"{'='*60}\n")
    
    FASTA_FILE = f"generated_sequences_fnet_bs{BATCH_SIZE}_{TIMESTAMP}.fasta"
    LOG_FILE = f"generation_log_fnet_bs{BATCH_SIZE}_{TIMESTAMP}.csv"
    
    print(f"Output FASTA: {FASTA_FILE}")
    print(f"Output Log: {LOG_FILE}\n")

    csv_header = [
        'seq_id', 'batch_size', 'sequence_length', 'generation_time_sec', 
        'time_per_seq_sec', 'avg_iter_latency_ms', 'throughput_total_tokens_per_sec',
        'throughput_per_seq_tokens_per_sec', 'memory_usage_mb', 'perplexity',
        'seed', 'temperature', 'top_k', 'top_p', 'num_iterations'
    ]

    all_memory_readings = []
    sequences_generated = 0

    with open(FASTA_FILE, 'w') as fasta_out, open(LOG_FILE, 'w', newline='') as csv_out:
        csv_writer = csv.DictWriter(csv_out, fieldnames=csv_header)
        csv_writer.writeheader()
        
        # Calculate number of batches needed
        num_batches = (NUM_SEQUENCES + BATCH_SIZE - 1) // BATCH_SIZE
        
        for batch_idx in tqdm(range(num_batches), desc=f"Generating (BS={BATCH_SIZE})"):
            # Handle last batch (might be smaller)
            current_batch_size = min(BATCH_SIZE, NUM_SEQUENCES - sequences_generated)
            if current_batch_size <= 0:
                break
            
            # Track memory BEFORE generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(DEVICE)
                initial_memory = torch.cuda.memory_allocated(DEVICE) / 1024**2  # MB
            
            # Generate batch of sequences (timed)
            sequences, total_time, time_per_seq, avg_iter_latency, throughput_total, throughput_per_seq, actual_len = generate_sequence_batch(
                model, tokenizer, SEQ_LENGTH, NUM_ITERATIONS, TEMPERATURE, TOP_K, TOP_P, current_batch_size
            )

            # Track memory AFTER generation
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated(DEVICE) / 1024**2
                memory_used = peak_memory - initial_memory
            else:
                memory_used = 0.0
            
            all_memory_readings.append(memory_used)

            # Process each sequence in the batch
            for seq_idx, seq in enumerate(sequences):
                seq_id = f"seq_{sequences_generated + seq_idx + 1:05d}"
                
                # Calculate perplexity AFTER generation (not included in timing)
                perplexity = calculate_perplexity(model, tokenizer, seq)

                fasta_out.write(f">{seq_id}\n{seq}\n")

                csv_writer.writerow({
                    'seq_id': seq_id,
                    'batch_size': BATCH_SIZE,
                    'sequence_length': actual_len,
                    'generation_time_sec': round(total_time, 4),
                    'time_per_seq_sec': round(time_per_seq, 4),
                    'avg_iter_latency_ms': round(avg_iter_latency * 1000, 3),
                    'throughput_total_tokens_per_sec': round(throughput_total, 2),
                    'throughput_per_seq_tokens_per_sec': round(throughput_per_seq, 2),
                    'memory_usage_mb': round(memory_used, 2),
                    'perplexity': round(perplexity, 4),
                    'seed': RANDOM_SEED,
                    'temperature': TEMPERATURE,
                    'top_k': TOP_K,
                    'top_p': TOP_P,
                    'num_iterations': NUM_ITERATIONS
                })

            sequences_generated += current_batch_size
            fasta_out.flush()
            csv_out.flush()

    print(f"\n✓ Generated {sequences_generated} sequences with batch size {BATCH_SIZE}")
    print(f"✓ FASTA file saved: {FASTA_FILE}")
    print(f"✓ Log file saved: {LOG_FILE}")

    # Print memory statistics for this batch size
    if all_memory_readings:
        print(f"\n💾 Memory Statistics (Batch Size {BATCH_SIZE}):")
        print(f"  Mean: {statistics.mean(all_memory_readings):.2f} MB")
        print(f"  Median: {statistics.median(all_memory_readings):.2f} MB")
        print(f"  Std Dev: {statistics.stdev(all_memory_readings):.2f} MB")
        print(f"  Max: {max(all_memory_readings):.2f} MB")
        print(f"  Min: {min(all_memory_readings):.2f} MB")

print("\n" + "="*60)
print("All batch size tests complete!")
print("="*60)
