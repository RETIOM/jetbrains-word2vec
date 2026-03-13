import argparse
import numpy as np
from src.model.cbow import CBOWInference

def cosine_similarity(v, U):
    num = np.dot(U, v)
    den = np.linalg.norm(U, axis=1) * np.linalg.norm(v)
    return num / (den + 1e-9)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument("--adapter-path", type=str, required=False, default=None)
    parser.add_argument("--positive", type=str, nargs="+", default=["king", "woman"])
    parser.add_argument("--negative", type=str, nargs="+", default=["man"])
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    model = CBOWInference.from_file(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        adapter_path=args.adapter_path
    )

    try:
        pos_vecs = [model.embed(w)[0] for w in args.positive]
        neg_vecs = [model.embed(w)[0] for w in args.negative]
    except RuntimeError as e:
        print(e)
        return

    target_vec = sum(pos_vecs) - sum(neg_vecs)

    sims = cosine_similarity(target_vec, model.embeddings)
    top_indices = np.argsort(sims)[-args.top_k:][::-1]

    print(f"\n[Analogies]")
    print(f"   Original: {' + '.join(args.positive)} - {' - '.join(args.negative)}")
    print(f"--------------------------------------------------")
    print(f"   {'Word':<20} |   {'Cosine Similarity':<15}")
    print(f"--------------------------------------------------")
    for idx in top_indices:
        word = model.tokenizer.decode(idx)
        print(f"   {word:<20} |   {sims[idx]:.4f}")
    print(f"--------------------------------------------------\n")

if __name__ == "__main__":
    main()
