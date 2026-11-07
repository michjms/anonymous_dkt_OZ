import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import scipy.stats as stats
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

studentseq_oz = dfoz.groupby('student_id').apply(
    lambda x: x.sort_values('start_time')[['skills', 'correct']].values
)
studentseq_noz = dfnoz.groupby('student_id').apply(
    lambda x: x.sort_values('start_time')[['skills', 'correct']].values
)

all_skills = set()
for seq in studentseq_oz.values:
    if len(seq) > 0:
        all_skills.update(seq[:, 0])
for seq in studentseq_noz.values:
    if len(seq) > 0:
        all_skills.update(seq[:, 0])
all_skills.discard(None) 

skill_to_id = {skill: idx for idx, skill in enumerate(sorted(all_skills))}

def build_sequences(student_seqs, skill_to_id):
    sequences = []
    skipped = 0
    for seq in student_seqs.values:
        if len(seq) < 2:
            skipped += 1
            continue
        rows = []
        for skill_name, correct_val in seq:
            if skill_name is None or skill_name not in skill_to_id:
                continue
            try:
                c = int(float(correct_val))
            except (ValueError, TypeError):
                continue
            if c not in (0, 1):
                continue
            rows.append((skill_to_id[skill_name], c))
        if len(rows) >= 2:
            sequences.append(np.array(rows, dtype=int))
        else:
            skipped += 1
    return sequences

sequencesOZ = build_sequences(studentseq_oz, skill_to_id)
sequencesNOZ = build_sequences(studentseq_noz, skill_to_id)

skills_in_OZ = set()
for seq in sequencesOZ:
    skills_in_OZ.update(seq[:, 0].tolist())

skills_in_NOZ = set()
for seq in sequencesNOZ:
    skills_in_NOZ.update(seq[:, 0].tolist())

overlapping_ids = sorted(skills_in_OZ & skills_in_NOZ)
remap = {old: new for new, old in enumerate(overlapping_ids)}
num_skills = len(remap)

id_to_skill = {v: k for k, v in skill_to_id.items()}
for old_id in overlapping_ids:
    print(f"  {remap[old_id]:3d} <- {old_id:3d} : {id_to_skill[old_id]}")

def filter_and_remap(sequences, remap):
    valid_ids = set(remap.keys())
    result = []
    for seq in sequences:
        mask = np.isin(seq[:, 0], list(valid_ids))
        filtered = seq[mask]
        if len(filtered) < 2:
            continue
        new_skills = np.array([remap[s] for s in filtered[:, 0]])
        result.append(np.column_stack([new_skills, filtered[:, 1]]))
    return result

sequencesOZ_final = filter_and_remap(sequencesOZ, remap)
sequencesNOZ_final = filter_and_remap(sequencesNOZ, remap)

MAX_SEQ_LEN = 1000
def truncate_sequence(seq, max_len):
    if len(seq) > max_len:
        return seq[-max_len:]
    return seq