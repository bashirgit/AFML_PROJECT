import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from transformers import EsmTokenizer, EsmModel
import random
from scipy import stats

# ================================
# CONFIGURATION
# ================================
FILE_PATHS = [
    '/home/mluser/AFML_RISHABH/10k sequences/kinases_cluster_train_10k.fasta',
    '/home/mluser/fnet_generated_sequences_2000.fasta',
    '/home/mluser/progen_gen_2000.fasta',
    '/home/mluser/esm_generated_2000.fasta',
]

SAMPLE_SIZE = 200
OUTPUT_BASE = "/home/mluser/AFML_RISHABH/Output_Analysis_Final/Hybrid_plddt_detailed_outputs"
os.makedirs(OUTPUT_BASE, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚗️ Computing hybrid pLDDT proxies via ESM2 embeddings on {device}...")

# ================================
# LOAD ESM2 MODEL
# ================================
tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D").to(device).eval()

# ================================
# HYBRID pLDDT FUNCTION
# ================================
def esm2_confidence(sequences, batch_size=64):
    confs = []
    for i in tqdm(range(0, len(sequences), batch_size), desc="pLDDT proxy"):
        batch = sequences[i:i + batch_size]
        tokens = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        with torch.no_grad():
            out = model(**tokens)
            emb = out.last_hidden_state.mean(dim=1).detach().cpu().numpy()
            norms = np.linalg.norm(emb, axis=1)
            confs.extend(norms / np.max(norms))
    return np.array(confs)

# ================================
# ANALYSIS PIPELINE
# ================================
summary_data = []

for fasta_path in FILE_PATHS:
    dataset_name = os.path.splitext(os.path.basename(fasta_path))[0]
    print(f"\n📂 Processing: {dataset_name}")

    # Load sequences
    sequences = [str(rec.seq) for rec in SeqIO.parse(fasta_path, "fasta")]
    if len(sequences) > SAMPLE_SIZE:
        sequences = random.sample(sequences, SAMPLE_SIZE)

    # Compute hybrid pLDDT proxy
    confs = esm2_confidence(sequences)

    # Compute summary statistics
    mean_conf = np.mean(confs)
    sd_conf = np.std(confs)
    ci_low, ci_high = stats.t.interval(
        0.95, len(confs)-1, loc=mean_conf, scale=stats.sem(confs)
    )

    # Save detailed results
    df = pd.DataFrame({
        "Sequence_ID": [f"seq_{i+1}" for i in range(len(confs))],
        "Hybrid_pLDDT_Proxy": confs
    })
    out_path = os.path.join(OUTPUT_BASE, f"{dataset_name}_hybrid_plddt.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Saved detailed output: {out_path}")

    # Store summary
    summary_data.append({
        "Dataset": dataset_name,
        "N": len(confs),
        "Mean_Hybrid_pLDDT": round(mean_conf, 4),
        "SD": round(sd_conf, 4),
        "95%_CI_Low": round(ci_low, 4),
        "95%_CI_High": round(ci_high, 4)
    })

# ================================
# SAVE SUMMARY FILE
# ================================
summary_df = pd.DataFrame(summary_data)
summary_path = os.path.join(OUTPUT_BASE, "hybrid_plddt_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\n📊 Summary saved to: {summary_path}")
print(summary_df)