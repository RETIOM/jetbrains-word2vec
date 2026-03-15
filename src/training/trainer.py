import os
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
from argparse import Namespace

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv

from src.model.models import TrainableModel
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
    config: Namespace

    patience: int = field(init=False)
    total_steps: int = field(init=False)
    do_step: bool = field(init=False, default=False)
    best_loss: float = field(init=False, default=float("inf"))
    ticker: int = field(init=False)
    best_weights: list[np.ndarray] = field(init=False, default_factory=list)

    def __post_init__(self):
        self.patience = self.config.patience

    def fit(self, train_dataloader, val_dataloader) -> dict:
        metrics = defaultdict(list)
        train_loss: float = 0.0
        self.total_steps = (
            len(train_dataloader.dataset)
            if hasattr(train_dataloader.dataset, "__len__")
            else -1
        )
        try:
            epoch_bar = tqdm(
                range(self.config.epochs),
                desc="Training Pipeline",
                bar_format="{l_bar}{bar:40}{r_bar}",
            )
            for epoch in epoch_bar:
                self.do_step = self.config.lr_scheduling != "off" and (
                    epoch > 0 or self.total_steps != -1
                )

                train_loss, self.total_steps = self._train_epoch(epoch, train_dataloader)
                metrics["train_loss"].append(train_loss)

                if self.config.val_path is not None:
                    val_loss = self._validate_epoch(val_dataloader)
                    metrics["val_loss"].append(val_loss)
                    epoch_bar.set_postfix(
                        train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}"
                    )
                    if self._check_early_stopping(metrics["val_loss"][-1]):
                        break
                else:
                    epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}")
        finally:
            self._save_model(self.config.save_dir)

            if self.config.store_metrics:
                self._export_metrics(metrics, self.config.save_dir)

        print("\n[Training Summary]")
        print(f"   Epochs Completed: {len(metrics['train_loss'])}")
        print(
            f"   Final Train Loss: {metrics['train_loss'][-1]:.4f}"
            if metrics["train_loss"]
            else "   Final Train Loss: N/A"
        )
        if self.config.val_path is not None and metrics["val_loss"]:
            print(
                f"   Best Val Loss:    {self.best_loss:.4f} (Finished at {metrics['val_loss'][-1]:.4f})"
            )
        print()

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

    def _train_epoch(self, epoch: int, dataloader: Dataloader) -> tuple[float, int]:
        epoch_loss = 0.0
        num_batches = 0
        ema_loss = None

        p_bar = tqdm(
            dataloader,
            leave=False,
            bar_format="{l_bar}{bar:30}{r_bar}",
            desc="  ↳ Training  ",
        )

        for step, (batch, targets) in enumerate(p_bar):
            logits = self.model.forward(batch, targets)
            loss = self.loss.forward(logits, targets)

            ema_loss = loss if ema_loss is None else 0.9 * ema_loss + 0.1 * loss
            p_bar.set_postfix(loss=f"{ema_loss:.4f}", refresh=False)

            delta_output = self.loss.backward()
            self.model.backward(delta_output)
            self.optimizer.step(self.model.params(), self.model.grads())

            if self.do_step:
                self._step_learning_rate(epoch, step)

            num_batches += 1
            epoch_loss += loss

        return epoch_loss / max(1, num_batches), num_batches

    def _validate_epoch(self, val_dataloader: Dataloader) -> float:
        total_loss = 0.0
        num_batches = 0

        p_bar = tqdm(
            val_dataloader,
            leave=False,
            bar_format="{l_bar}{bar:30}{r_bar}",
            desc="  ↳ Validation",
        )

        for batch, targets in p_bar:
            logits = self.model.forward(batch, targets)
            loss = self.loss.forward(logits, targets)

            total_loss += loss
            num_batches += 1

            current_avg = total_loss / num_batches
            p_bar.set_postfix(avg_loss=f"{current_avg:.4f}", refresh=False)

        return total_loss / max(1, num_batches)

    def _check_early_stopping(self, val_loss: float) -> bool:
        if self.patience < 0:
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

    def _step_learning_rate(self, current_epoch: int, step_in_epoch: int) -> None:
        match self.config.lr_scheduling:
            case "linear":
                # Assuming self.total_steps is the steps per epoch
                if self.total_steps <= 0:
                    return

                # Calculate progress out of total expected training steps
                total_training_steps = self.total_steps * self.config.epochs
                global_step = (current_epoch * self.total_steps) + step_in_epoch
                progress = min(1.0, global_step / max(1, total_training_steps))
                
                new_lr = self.config.lr * (1 - progress)
                self.optimizer.lr = max(self.config.min_lr, new_lr)
            case _:
                raise RuntimeError(
                    f"Invalid learning rate scheduling specified: {self.config.lr_scheduling}"
                )

    def _save_model(self, save_dir: str) -> None:
        params = {}
        if self.best_weights:
            params["model_params"] = self.best_weights
        else:
            params["model_params"] = self.model.params()

        params["tokenizer"] = self.tokenizer

        self.model.save(params, save_dir)

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
        if self.config.plot:
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
