# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 18:00:35 2026

@author: azamb
"""

import os
import copy
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DKT(nn.Module):
    """
    Skill-only DKT implementation compatible with the existing project pipeline.

    This version follows the original DKT style more closely:
    - interaction inputs are integer-encoded with padding=0
    - one-hot inputs are built inside the model
    - model outputs logits
    - training uses BCEWithLogitsLoss
    """

    def __init__(self, n_skill, hidden_dim=100, num_layers=1, dropout=0.2):
        super().__init__()
        self.n_skill = n_skill
        self.input_size = 2 * n_skill + 1   # 0 reserved for padding
        self.output_size = n_skill

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(hidden_dim, self.output_size)

    def forward(self, skill_inputs, hidden=None):
        """
        skill_inputs: LongTensor [B, T]
            values in:
            - 0 for padding
            - 2*s + 1 for incorrect response on skill s
            - 2*s + 2 for correct response on skill s
        """
        x = F.one_hot(skill_inputs, num_classes=self.input_size).float()
        output, hidden = self.lstm(x, hx=hidden)
        output = self.dropout(output)
        logits = self.out(output)   # [B, T, n_skill]
        return logits, hidden

    def repackage_hidden(self, hidden):
        return tuple(v.detach() for v in hidden)


def _read_dkt_sequences(path, n_skill):
    """
    Reads one file or a list of files in the project's 4-line sequential format.

    For each student sequence of length L:
      input:  interactions for positions 0..L-2
      target: correctness for positions 1..L-1
      target skill: skills for positions 1..L-1

    Returns:
        list of tuples:
            (skill_inputs, next_skill_ids, next_labels)
    """
    if isinstance(path, (str, os.PathLike)):
        paths = [path]
    else:
        paths = list(path)

    sequences = []

    for one_path in paths:
        with open(one_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        if len(lines) % 4 != 0:
            raise ValueError(f"Invalid sequential file format in {one_path}")

        for i in range(0, len(lines), 4):
            skills = [int(x) for x in lines[i + 1].split(",") if x != ""]
            answers = [int(x) for x in lines[i + 3].split(",") if x != ""]

            if len(skills) != len(answers):
                raise ValueError(
                    f"Skill/answer length mismatch in {one_path}, block {i // 4}"
                )

            if len(skills) < 2:
                continue

            skill_inputs = []
            next_skill_ids = []
            next_labels = []

            for t in range(len(skills) - 1):
                s_t = skills[t]
                a_t = answers[t]

                if s_t < 0 or s_t >= n_skill:
                    raise ValueError(f"Skill id {s_t} out of range for n_skill={n_skill}")

                # Original-style interaction encoding:
                # padding = 0
                # incorrect on skill s -> 2*s + 1
                # correct   on skill s -> 2*s + 2
                interaction_id = 2 * s_t + int(a_t) + 1

                skill_inputs.append(interaction_id)
                next_skill_ids.append(skills[t + 1])
                next_labels.append(answers[t + 1])

            sequences.append((
                torch.tensor(skill_inputs, dtype=torch.long),
                torch.tensor(next_skill_ids, dtype=torch.long),
                torch.tensor(next_labels, dtype=torch.float32)
            ))

    return sequences


def _prepare_batches(data, batch_size, shuffle=True):
    """
    Creates padded batches.

    Returns list of tuples:
        skill_inputs: [B, T] long, padded with 0
        next_skill_ids: [B, T] long, padded with 0
        next_labels: [B, T] float, padded with -1
    """
    if shuffle:
        random.shuffle(data)

    batches = []
    for k in range(0, len(data), batch_size):
        batch = data[k:k + batch_size]
        skill_inputs_list, next_skill_ids_list, next_labels_list = zip(*batch)

        skill_inputs = pad_sequence(skill_inputs_list, batch_first=True, padding_value=0)
        next_skill_ids = pad_sequence(next_skill_ids_list, batch_first=True, padding_value=0)
        next_labels = pad_sequence(next_labels_list, batch_first=True, padding_value=-1)

        batches.append((
            skill_inputs.to(device),
            next_skill_ids.to(device),
            next_labels.to(device)
        ))

    return batches


def _get_target_preds(logits, next_skill_ids, next_labels):
    """
    logits: [B, T, n_skill]
    next_skill_ids: [B, T]
    next_labels: [B, T]

    Returns:
        pred_logits_valid: [N_valid]
        labels_valid: [N_valid]
    """
    mask = next_labels >= 0
    gathered = logits.gather(2, next_skill_ids.unsqueeze(-1)).squeeze(-1)
    pred_logits_valid = gathered[mask]
    labels_valid = next_labels[mask]
    return pred_logits_valid, labels_valid


def _evaluate_dkt(model, data, batch_size):
    criterion = nn.BCEWithLogitsLoss()
    model.eval()

    all_probs = []
    all_labels = []
    losses = []

    batches = _prepare_batches(data, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for skill_inputs, next_skill_ids, next_labels in batches:
            logits, _ = model(skill_inputs)
            pred_logits, labels = _get_target_preds(logits, next_skill_ids, next_labels)

            loss = criterion(pred_logits, labels)
            losses.append(loss.item())

            probs = torch.sigmoid(pred_logits)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    if len(all_probs) == 0:
        return float("nan"), float("nan"), float("nan"), [], []

    rmse = math.sqrt(metrics.mean_squared_error(all_labels, all_probs))
    acc = ((torch.tensor(all_probs) >= 0.5) == (torch.tensor(all_labels) == 1)).float().mean().item()

    if len(set(all_labels)) < 2:
        auc = float("nan")
    else:
        auc = metrics.roc_auc_score(all_labels, all_probs)

    return auc, rmse, acc, all_probs, all_labels


def train_DKT(train_path, valid_path, n_skill,
              hidden_dim=100, num_layers=1, dropout=0.2,
              lr=1e-3, batch_size=32, epochs=30):
    train_data = _read_dkt_sequences(train_path, n_skill)
    valid_data = _read_dkt_sequences(valid_path, n_skill)

    model = DKT(
        n_skill=n_skill,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = -float("inf")
    best_state_dict = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        train_batches = _prepare_batches(train_data, batch_size=batch_size, shuffle=True)
        running_loss = 0.0

        train_bar = tqdm(train_batches, desc=f"DKT train epoch [{epoch + 1}/{epochs}]")
        for skill_inputs, next_skill_ids, next_labels in train_bar:
            logits, _ = model(skill_inputs)
            pred_logits, labels = _get_target_preds(logits, next_skill_ids, next_labels)

            if labels.numel() == 0:
                continue

            loss = criterion(pred_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = running_loss / max(len(train_batches), 1)

        val_auc, val_rmse, val_acc, _, _ = _evaluate_dkt(
            model=model,
            data=valid_data,
            batch_size=batch_size
        )

        print(
            f"[epoch {epoch + 1}] "
            f"train_loss: {avg_train_loss:.4f} "
            f"val_auc: {val_auc:.4f} "
            f"val_rmse: {val_rmse:.4f} "
            f"val_acc: {val_acc:.4f}"
        )

        if not math.isnan(val_auc):
            if val_auc > best_auc:
                best_auc = val_auc
                best_state_dict = copy.deepcopy(model.state_dict())
        elif best_auc == -float("inf"):
            # fallback in case validation labels contain only one class
            best_state_dict = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state_dict)
    return model


def train_predict_DKT(train_path, valid_path, test_path,
                      n_skill, hidden_dim=100, num_layers=1,
                      dropout=0.2, lr=1e-3, batch_size=32, epochs=30):
    """
    Trains DKT and returns:
        preds: list of probabilities for next-step correctness
        labels: list of corresponding true labels

    This output format matches your existing pipeline and evaluation code.
    """
    model = train_DKT(
        train_path=train_path,
        valid_path=valid_path,
        n_skill=n_skill,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs
    )

    test_data = _read_dkt_sequences(test_path, n_skill)
    _, _, _, preds, labels = _evaluate_dkt(
        model=model,
        data=test_data,
        batch_size=batch_size
    )

    return preds, labels