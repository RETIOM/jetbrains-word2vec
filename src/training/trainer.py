import os
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv

from src.model.model import TrainableModel
from src.model.loss import Loss
from src.model.optimizer import Optimizer
from src.data.dataloader import Dataloader
from src.data.tokenizer import Tokenizer


@dataclass(slots=True)
class Trainer:
    model: TrainableModel
    loss: Loss
    optimizer: Optimizer
    tokenizer: Tokenizer
    config: dict

    patience: int = field(init=False)
    best_loss: float = field(init=False, default=float("inf"))
    ticker: int = field(init=False)
    best_weights: list[np.ndarray] = field(init=False, default_factory=list)

    def __post_init__(self):
        self.patience = self.config.get("patience", 0)

    def fit(self, train_dataloader, val_dataloader) -> dict:
        metrics = defaultdict(list)
        try:
            for epoch in tqdm(range(self.config.get("num_epochs", 10)), desc="EPOCH"):
                metrics["train_loss"].append(self._train_epoch(train_dataloader))
                if self.config.get("val"):
                    metrics["val_loss"].append(self._validate_epoch(val_dataloader))
                    if self._check_early_stopping(metrics["val_loss"][-1]):
                        break
        finally:
            self._save_model(self.config.get("save_dir", "."))

            if self.config.get("save_metrics", None):
                self._export_metrics(metrics, self.config.get("save_dir", "."))

        return metrics

    def test(self, test_dataloader: Dataloader) -> float:
        total_loss = 0.0
        num_batches = 0
        for batch, targets in tqdm(test_dataloader):
            logits = self.model.forward(batch, targets)
            loss = self.loss.forward(logits, targets)
            total_loss += loss
            num_batches += 1

        return total_loss / max(1, num_batches)

    def _train_epoch(self, dataloader: Dataloader) -> float:
        epoch_loss = 0.0
        num_batches = 0
        p_bar = tqdm(dataloader, leave=False)
        for batch, targets in p_bar:
            logits = self.model.forward(batch, targets)
            loss = self.loss.forward(logits, targets)
            p_bar.set_description(f"LOSS: {loss}")
            delta_output = self.loss.backward()
            self.model.backward(delta_output)
            self.optimizer.step(self.model.params(), self.model.grads())
            num_batches += 1
            epoch_loss += loss

        return epoch_loss / max(1, num_batches)

    def _validate_epoch(self, val_dataloader: Dataloader) -> float:
        total_loss = 0.0
        num_batches = 0
        p_bar = tqdm(val_dataloader, desc=f"VALIDATION, LOSS: {0}")
        for batch, targets in p_bar:
            logits = self.model.forward(batch, targets)
            loss = self.loss.forward(logits, targets)
            p_bar.set_description(f"VALIDATION, LOSS: {loss}")
            total_loss += loss
            num_batches += 1

        return total_loss / max(1, num_batches)

    def _check_early_stopping(self, val_loss: float) -> bool:
        if self.patience <= 0:
            return False

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.ticker = 0
            self.best_weights = [np.copy(p) for p in self.model.params()]
            return False

        self.ticker += 1

        if self.ticker >= self.patience:
            print(f"EARLY STOP - NO IMPROVEMENT FOR {self.patience} EPOCHS")
            return True

        return False

    def _save_model(self, save_dir: str) -> None:

        os.makedirs(os.path.dirname(save_dir), exist_ok=True)

        if getattr(self, "best_weights", None) is not None:
            weights_to_save = self.best_weights
        else:
            weights_to_save = self.model.params()

        embeddings = weights_to_save[0]
        w_out = weights_to_save[1] if len(weights_to_save) > 1 else np.array([])

        word2idx = self.tokenizer.word2idx
        idx2word = self.tokenizer.idx2word

        np.savez(
            Path(save_dir) / "best_model.npz",
            embeddings=embeddings,
            w_out=w_out,
            word2idx=np.array(word2idx, dtype=object),
            idx2word=np.array(idx2word, dtype=object),
        )

    def _export_metrics(self, metrics: dict, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)

        if "train_loss" not in metrics or not metrics["train_loss"]:
            return

        num_epochs = len(metrics["train_loss"])
        epochs = list(range(num_epochs))

        csv_path = Path(save_dir) / "metrics.csv"
        plot_path = Path(save_dir) / "loss.png"
        with open(csv_path, "w", newline="") as f:
            headers = ["epoch"] + list(metrics.keys())
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

            for i in range(num_epochs):
                row = {"epoch": i}
                for key, values in metrics.items():
                    if i >= len(values):
                        break
                    row[key] = values[i]
                writer.writerow(row)
        if self.config.get("plot", False):
            plt.figure(figsize=(10, 6))

            plt.plot(
                epochs,
                metrics["train_loss"],
                label="Train Loss",
                color="blue",
                marker="o",
                linewidth=2,
            )

            if "val_loss" in metrics and metrics["val_loss"]:
                val_epochs = list(range(len(metrics["val_loss"])))
                plt.plot(
                    val_epochs,
                    metrics["val_loss"],
                    label="Validation Loss",
                    color="orange",
                    marker="x",
                    linewidth=2,
                )
            plt.title("Training & Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend()

            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
