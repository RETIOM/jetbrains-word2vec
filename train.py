import argparse
import random
from functools import partial

import numpy as np

from src.data.dataset import WikiTextDataset
from src.data.dataloader import IterDataloader
from src.utils.collate import wikitext_collate_fn
from src.model.cbow import CBOWEmbedder, CBOW
from src.model.adapter import NegativeSamplingHead
from src.model.optimizer import SGD
from src.model.loss import NegativeSamplingLoss
from src.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        prog="CBOW Training",
        description="Launches training for a CBOW word2vec model through a (hopefully) universal-ish framework",
    )

    parser.add_argument(
        "--train-path",
        type=str,
        required=True,
        help="Path to the a text file containig training data",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Path where model, tokenizer, adapter weights will be stored. Also contains metrics if they are to be collected.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        required=False,
        default=64,
        help="Number of samples per batch during training",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        required=False,
        default=10,
        help="Maximum number of epochs to train",
    )

    parser.add_argument(
        "--lr",
        type=float,
        required=False,
        default=0.01,
        help="Learning rate for the optimizer",
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        required=False,
        default=100,
        help="Dimensionality of word embeddings",
    )

    parser.add_argument(
        "--window-size",
        type=int,
        required=False,
        default=5,
        help="Context window size for CBOW",
    )

    parser.add_argument(
        "--dynamic-window",
        action="store_true",
        help="Whether or not to use dynamic window size in training",
    )

    parser.add_argument(
        "--negative-samples",
        type=int,
        required=False,
        default=5,
        help="Number of negative samples for negative sampling",
    )

    parser.add_argument(
        "--min-count",
        type=int,
        required=False,
        default=5,
        help="Minimum frequency for words to be included in the vocabulary",
    )

    parser.add_argument(
        "--subsampling-rate",
        type=float,
        required=False,
        default=1e-3,
        help="Subsampling rate for frequent words",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        required=False,
        default="SGD",
        choices=["SGD"],
        help="Optimizer used for training",
    )

    parser.add_argument(
        "--val-path",
        type=str,
        required=False,
        default=None,
        help="Path to the text file containing validation data",
    )

    parser.add_argument(
        "--patience",
        type=int,
        required=False,
        default=0,
        help="Num. epochs of no improvement before early stop. Note: Only available if --val-path is specified",
    )

    parser.add_argument(
        "--store-metrics", action="store_true", help="Enables metric saving"
    )

    parser.add_argument("--plot", action="store_true", help="Enables metric plotting")

    parser.add_argument(
        "--seed", required=False, default=sum(ord(c) for c in "jetbrains")
    )

    return parser.parse_args()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)


def load_data(args):

    train_dataset = WikiTextDataset(
        data_path=args.train_path,
        window_size=args.window_size,
        use_dynamic_window=args.dynamic_window,
        min_count=args.min_count,
        subsampling_threshold=args.subsampling_rate,
    )

    val_dataset = None
    if args.val_path:
        val_dataset = WikiTextDataset(
            data_path=args.val_path,
            window_size=args.window_size,
            subsampling_threshold=args.subsampling_rate,
            tokenizer=train_dataset.tokenizer,
        )

    return train_dataset, val_dataset


def build_dataloaders(train_dataset, val_dataset, args):
    collate_fn = partial(wikitext_collate_fn, window_size=args.window_size)
    train_loader = IterDataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
    )
    val_loader = None
    if val_dataset:
        val_loader = IterDataloader(
            dataset=val_dataset,
            batch_size=1,
            collate_fn=collate_fn,
        )
    return train_loader, val_loader


def build_model(args, data):
    embedder = CBOWEmbedder(
        vocab_size=data.tokenizer.vocab_size, n_dim=args.embedding_dim
    )
    head = NegativeSamplingHead(
        in_dim=embedder.embeddings.shape[1],
        vocab_size=data.tokenizer.vocab_size,
        word_counts=data.word_counts,
        k=args.negative_samples,
    )
    model = CBOW(embedder=embedder, head=head)
    optimizer = None
    match args.optimizer:
        case "SGD":
            optimizer = SGD(lr=args.lr)
        case _:
            raise ValueError("Optimizer not implemented")
    loss_fn = NegativeSamplingLoss()

    return model, loss_fn, optimizer


def train(model, loss, optimizer, tokenizer, train_dataloader, val_dataloader, args):
    trainer = Trainer(
        model=model, loss=loss, optimizer=optimizer, tokenizer=tokenizer, config=args
    )
    metrics = trainer.fit(
        train_dataloader=train_dataloader, val_dataloader=val_dataloader
    )
    return metrics


def main():
    args = parse_args()
    set_seed(args)
    train_dataset, val_dataset = load_data(args=args)
    tokenizer = train_dataset.tokenizer
    train_dataloader, val_dataloader = build_dataloaders(
        train_dataset=train_dataset, val_dataset=val_dataset, args=args
    )
    model, loss, optimizer = build_model(args=args, data=train_dataset)
    train(
        model=model,
        loss=loss,
        optimizer=optimizer,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        args=args,
    )


if __name__ == "__main__":
    main()
