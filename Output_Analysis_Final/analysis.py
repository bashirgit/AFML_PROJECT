#!/usr/bin/env python3
"""
Protein Sequence Evaluation (No pLDDT Version)
----------------------------------------------
Evaluates generated protein sequences against a training set on:
- Amino acid composition
- K-mer (k=3,5) Jensen-Shannon Divergence
- Shannon entropy
- Hydrophobicity
- Net charge
- Molecular weight
- Repetitive content
- Low-complexity fraction
- Motif presence

Outputs:
- Per-sequence metrics CSVs for each dataset
- Summary statistics (mean ± 95% CI) CSV
- Violin plots for key metrics
"""

import os
import re
import json
import time
import random
import warnings
from collections import Counter
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, sem
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
warnings.filterwarnings('ignore')

STANDARD_AA = list("ACDEFGHIKLMNPQRSTVWY")

FILE_PATHS = {
    'train': '/home/mluser/AFML_RISHABH/10k sequences/kinases_cluster_train_10k.fasta',
    'fnet': '/home/mluser/fnet_generated_sequences_2000.fasta',
    'progen': '/home/mluser/progen_gen_2000.fasta',
    'esm': '/home/mluser/esm_generated_2000.fasta',
}

OUTPUT_DIR = '/home/mluser/AFML_RISHABH/Output_Analysis_Final'
TRAIN_SAMPLE_SIZE = 2000
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

# -----------------------------------------------------------------------------
# UTILS
# -----------------------------------------------------------------------------
def confidence_interval_95(data):
    if len(data) < 2:
        return (np.nan, np.nan)
    mean = np.mean(data)
    se = sem(data)
    ci = se * 1.96
    return (mean - ci, mean + ci)

def load_fasta(filepath: str) -> List[Tuple[str, str]]:
    seqs = []
    for rec in SeqIO.parse(filepath, "fasta"):
        seq = str(rec.seq).upper().replace("U", "X").replace("*", "")
        seq = ''.join([c for c in seq if c in "ACDEFGHIKLMNPQRSTVWYX"])
        if len(seq) > 0:
            seqs.append((rec.id, seq))
    return seqs

def amino_acid_composition(seqs: List[str]) -> Dict[str, float]:
    total = 0
    counter = Counter()
    for s in seqs:
        counter.update([c for c in s if c in STANDARD_AA])
        total += len([c for c in s if c in STANDARD_AA])
    if total == 0:
        return {aa: 0.0 for aa in STANDARD_AA}
    return {aa: counter.get(aa, 0) / total for aa in STANDARD_AA}

def kmer_counts(seq: str, k: int) -> Counter:
    c = Counter()
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if all(aa in STANDARD_AA for aa in kmer):
            c[kmer] += 1
    return c

def aggregate_kmer_distribution(seqs: List[str], k: int) -> Dict[str, float]:
    total_counter = Counter()
    total = 0
    for s in seqs:
        c = kmer_counts(s, k)
        total_counter.update(c)
        total += sum(c.values())
    if total == 0:
        return {}
    return {kmer: cnt / total for kmer, cnt in total_counter.items()}

def jensen_shannon_divergence(dist_p: Dict[str, float], dist_q: Dict[str, float]) -> float:
    keys = sorted(set(dist_p.keys()) | set(dist_q.keys()))
    if not keys:
        return float('nan')
    p = np.array([dist_p.get(k, 0.0) for k in keys], dtype=float)
    q = np.array([dist_q.get(k, 0.0) for k in keys], dtype=float)
    p = p / (p.sum() + 1e-10)
    q = q / (q.sum() + 1e-10)
    js = jensenshannon(p, q, base=2.0)
    return js**2

def shannon_entropy_of_sequence(seq: str) -> float:
    counts = Counter([c for c in seq if c in STANDARD_AA])
    if not counts:
        return 0.0
    probs = np.array(list(counts.values()), dtype=float)
    probs = probs / probs.sum()
    return float(entropy(probs, base=2))

def hydrophobicity_score(seq: str) -> float:
    kd = {'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,'I':4.5,'K':-3.9,'L':3.8,
           'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,'R':-4.5,'S':-0.8,'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3}
    vals = [kd.get(a, 0) for a in seq if a in kd]
    return np.mean(vals) if vals else 0.0

def charge_at_ph7(seq: str) -> float:
    pos = sum(1 for a in seq if a in 'KRH')
    neg = sum(1 for a in seq if a in 'DE')
    return (pos - neg) / len(seq) if len(seq) > 0 else 0.0

def molecular_weight(seq: str) -> float:
    mw = {'A':89,'C':121,'D':133,'E':147,'F':165,'G':75,'H':155,'I':131,'K':146,'L':131,
          'M':149,'N':132,'P':115,'Q':146,'R':174,'S':105,'T':119,'V':117,'W':204,'Y':181}
    return sum(mw.get(a,0) for a in seq)

def repetitive_content(seq: str, window: int = 3) -> float:
    repeats = 0
    for i in range(len(seq) - window + 1):
        if len(set(seq[i:i+window])) == 1:
            repeats += 1
    return repeats / max(1, len(seq) - window + 1)

def low_complexity_regions(seq: str, window: int = 12) -> float:
    low = 0
    for i in range(len(seq) - window + 1):
        subseq = seq[i:i+window]
        if shannon_entropy_of_sequence(subseq) < 2.0:
            low += 1
    return low / max(1, len(seq) - window + 1)

def motif_presence(seq: str, motif_regex: str) -> bool:
    try:
        return re.search(motif_regex, seq) is not None
    except:
        return False

# -----------------------------------------------------------------------------
# EVALUATION
# -----------------------------------------------------------------------------
def evaluate_model_sequences(model_name: str, seqs, refseqs, motifs=None) -> pd.DataFrame:
    ref_aa = amino_acid_composition(refseqs)
    ref_k3 = aggregate_kmer_distribution(refseqs, 3)
    ref_k5 = aggregate_kmer_distribution(refseqs, 5)
    
    results = []
    log(f"▶ Evaluating {model_name.upper()} ({len(seqs)} sequences)...")
    start_time = time.time()

    for sid, s in tqdm(seqs, desc=model_name):
        aa_freq = amino_acid_composition([s])
        l1 = sum(abs(aa_freq.get(k, 0)-ref_aa.get(k,0)) for k in STANDARD_AA)
        k3 = aggregate_kmer_distribution([s], 3)
        k5 = aggregate_kmer_distribution([s], 5)
        
        row = {
            'model': model_name,
            'sequence_id': sid,
            'length': len(s),
            'aa_L1_divergence': l1,
            'kmer3_JSD': jensen_shannon_divergence(k3, ref_k3),
            'kmer5_JSD': jensen_shannon_divergence(k5, ref_k5),
            'shannon_entropy': shannon_entropy_of_sequence(s),
            'hydrophobicity': hydrophobicity_score(s),
            'net_charge': charge_at_ph7(s),
            'molecular_weight': molecular_weight(s),
            'repetitive_content': repetitive_content(s),
            'low_complexity_fraction': low_complexity_regions(s)
        }
        if motifs:
            for name, regex in motifs.items():
                row[f'motif_{name}'] = int(motif_presence(s, regex))
        results.append(row)

    log(f"✓ Completed {model_name} in {time.time() - start_time:.2f}s")
    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# STATS + PLOTS
# -----------------------------------------------------------------------------
def compute_statistics_with_ci(df, metric):
    data = df[metric].dropna()
    if len(data) < 2:
        return {'count': len(data), 'mean': np.nan}
    ci_low, ci_high = confidence_interval_95(data)
    return {
        'count': len(data),
        'mean': np.mean(data),
        'sd': np.std(data),
        'sem': sem(data),
        'ci_lower': ci_low, 'ci_upper': ci_high
    }

def plot_distributions_with_stats(dfs, metric, outpath, reference='train'):
    fig, ax = plt.subplots(figsize=(7,5))
    order = [reference]+[m for m in dfs.keys() if m!=reference]
    colors = {'train':'#96CEB4','fnet':'#FF6B6B','progen':'#4ECDC4','esm':'#45B7D1'}
    data = []
    for m in order:
        if metric in dfs[m].columns:
            x = dfs[m][metric].dropna()
            data.append(pd.DataFrame({'model':m,'value':x}))
    df_all = pd.concat(data)
    sns.violinplot(x='model', y='value', data=df_all, palette=colors, ax=ax)
    ax.set_title(metric)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    log("=== FAST PROTEIN SEQUENCE EVALUATION (pLDDT-Free) ===")
    sequences = {m: load_fasta(f) for m, f in FILE_PATHS.items() if os.path.exists(f)}
    log(f"Loaded datasets: {list(sequences.keys())}")

    if 'train' not in sequences:
        log("ERROR: Missing training FASTA. Exiting.")
        return

    # --- Sample training sequences ---
    train = sequences['train']
    original_train_size = len(train)
    if len(train) > TRAIN_SAMPLE_SIZE:
        idx = np.random.choice(len(train), TRAIN_SAMPLE_SIZE, replace=False)
        train = [train[i] for i in idx]
        log(f"Sampled {TRAIN_SAMPLE_SIZE} sequences from training set (out of {original_train_size}).")

    # Reference sequences for comparative metrics
    refseqs = [s for _, s in train]

    # Motif regex definitions
    motifs = {
    'VAIK': r'V[A-Z]{0,1}IK',
    'HRD': r'HRD',
    'DFG': r'DFG',
    'APE': r'APE',
    'GxGxxG': r'G.G..G'
    }

    # --- Prepare datasets to evaluate ---
    dfs = {}
    datasets_to_eval = {'train': train}  # use the sampled train
    for name, seqs in sequences.items():
        if name != 'train':
            datasets_to_eval[name] = seqs

    # --- Evaluation loop ---
    for name, seqs in datasets_to_eval.items():
        log(f"▶ Evaluating {name.upper()} ({len(seqs)} sequences)...")
        df = evaluate_model_sequences(name, seqs, refseqs, motifs)
        dfs[name] = df
        out_csv = os.path.join(OUTPUT_DIR, f"{name}_metrics.csv")
        df.to_csv(out_csv, index=False)
        log(f"  Saved per-sequence metrics: {out_csv}")

    # --- Summary statistics ---
    log("Computing summary statistics...")
    summary = []
    metrics = [
        'length', 'aa_L1_divergence', 'kmer3_JSD', 'kmer5_JSD',
        'shannon_entropy', 'hydrophobicity', 'net_charge',
        'molecular_weight', 'repetitive_content', 'low_complexity_fraction'
    ]

    for mname, df in dfs.items():
        for metric in metrics:
            stats = compute_statistics_with_ci(df, metric)
            stats['model'] = mname
            stats['metric'] = metric
            summary.append(stats)

    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(OUTPUT_DIR, "summary_statistics.csv")
    summary_df.to_csv(summary_path, index=False)
    log(f"Saved summary statistics: {summary_path}")

    # --- Plots ---
    log("Generating violin plots...")
    for metric in metrics:
        out = os.path.join(OUTPUT_DIR, f"{metric}_plot.png")
        plot_distributions_with_stats(dfs, metric, out)
        log(f"  Saved: {out}")

    log("=== ALL ANALYSES COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
