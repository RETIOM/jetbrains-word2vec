# PyGlowstick - Torch's weird brother


## Task - JetBrains word2vec

Implement the core training loop of word2vec in pure NumPy (no PyTorch / TensorFlow or other ML frameworks). The applicant is free to choose any suitable text dataset. The task is to implement the optimization procedure (forward pass, loss, gradients, and parameter updates) for a standard word2vec variant (e.g. skip-gram with negative sampling or CBOW).

## Comment

As you can see I've gone a little bit overboard on the implementation of this project but I can explain.

I've been wanting to do something like this for a while but I couldn't get around to actually following through on that idea, there were always more urgent matters at hand, like actually getting a good night's sleep. This task has given me the motivation to actually go forth and implement a (*hopefully*) **fully modular framework**(like PyTorch's weird brother) for model training. It was a great opportunity to gain a deeper understanding of the technologies I use daily, which are usually buried under layers of abstraction.

Of course it wasn't all sunshine and roses, there was quite a bit of struggle(huge understatement) in the process. Some notable mentions:
- countless refactors as I was slowly figuring out the actual PLAN for the implementation (was mostly going of vibes to be honset) and fighitng for agnosticity(if that's a word) between components(led to the creation of adapter heads and many more weird components)
- no learning - after finishing the basic implementation it was a struggle to actually get the model to learn
    - "vanishing gradients" (wrong initialization of weight matricies) - thought it was the loss, so implemented negative sampling
    - `lr` scaling with batch_size(it took a while to figure out)
- slow learning - was always performing dense operations(went from 15 it/s to around 300 it/s by switching to sparse)

I hope you enjoy and thanks for the opportunity!

## Project structure

```text
.
├── train.py                 # Described below
├── demo.py                  # -||-
├── download_dataset.py      # Downloads the dataset I used
├── requirements.txt         
├── pyproject.toml           
├── data/                    
├── outputs/                 # Contains train logs and a model trained on the whole corpus           
├── notebooks/               
└── src/
    ├── data/
    │   ├── dataloader.py    # IterDataloader for __iter__ datasets
    │   ├── dataset.py       # Handles creating windows, subsampling, dynamic window
    │   └── tokenizer.py     
    ├── model/
    │   ├── adapter.py       # Adapter heads - they allow the model to be loss agnostic
    │   ├── cbow.py          # Core CBOW-related model implementations
    │   ├── inference.py     # Abstract classes for inference
    │   ├── loss.py          
    │   ├── models.py        # Abstract classes for model definition and training
    │   └── optimizer.py     
    ├── training/
    │   └── trainer.py       # Training orchestration(also handles validation, early stopping, ...)
    └── utils/
        ├── collate.py       # Custom collate for correct padding of edge(literal) words
        └── math_utils.py    
```


## Usage

Python: 3.14

Before using make sure to build and source the `.venv`:
```bash
# Using uv
uv sync
source .venv/bin/activate

# Using pip
python3.14 -m venv .venv
source .venv/bin/activate # Linux version
pip install -r requirements.txt
```

Also make sure you have data:
```bash
python3 download_dataset.py
```

NOTE:
By default only the _small_ model(trained using "Quick test") is bundled with git, while the _large_ model(trained on the full corpra) requires:
```bash
git lfs install
git lfs pull
```

**`train.py`**: Orchestrates the model training loop and saves weights. Run `python train.py -h` for more info.
- Quick test: 
```bash
python train.py \
    --train-path ./data/wikitext-103/wiki.test.tokens \
    --save-dir ./outputs/model \
    --batch-size 64 \
    --epochs 3 \
    --lr 0.025 \
    --embedding-dim 100 \
    --window-size 2 \
    --min-count 5 \
    --dynamic-window \
    --store-metrics \
    --plot
```
- Full train: 
```bash
python train.py \
    --train-path ./data/wikitext-103/wiki.train.tokens \
    --val-path ./data/wikitext-103/wiki.valid.tokens \
    --save-dir ./outputs/model \
    --batch-size 128 \
    --epochs 10 \
    --lr 0.025 \
    --min-lr 0.0005 \
    --lr-scheduling linear \
    --embedding-dim 300 \
    --window-size 5 \
    --negative-samples 10 \
    --subsampling-rate 0.00005 \
    --optimizer SGD \
    --patience 5 \
    --min-count 5 \
    --dynamic-window \
    --store-metrics \
    --plot
```
_dataset load might take a while_(around 1min for me)

**`demo.py`**: Demo of inference; Computes operations such as `king - man + woman` and finds similar words in the vocab
- Example: 
```bash
python demo.py \
    --model-path ./outputs/model/small/embedder.npz \
    --tokenizer-path ./outputs/model/small/tokenizer.npz \
    --positive king woman \
    --negative man \
    --top-k 10
```
_to inference with large be sure to pull it(instruction above) and change paths_

