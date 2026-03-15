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
    parser.add_argument(
        "--include-inputs", action="store_true", help="Include input words in results"
    )
    args = parser.parse_args()

    model = CBOWInference.from_file(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        adapter_path=args.adapter_path,
    )

    try:
        pos_vecs = [model.embed(w)[0] for w in args.positive]
        neg_vecs = [model.embed(w)[0] for w in args.negative]
    except RuntimeError as e:
        print(e)
        return

    target_vec = np.sum(pos_vecs, axis=0) - np.sum(neg_vecs, axis=0)
    sims = cosine_similarity(target_vec, model.embeddings)

    if not args.include_inputs:
        input_words = args.positive + args.negative
        for word in input_words:
            try:
                vec = model.embed(word)[0]
                idx = np.where(np.all(model.embeddings == vec, axis=1))[0]
                if len(idx) > 0:
                    sims[idx[0]] = -np.inf
            except:
                pass

    top_indices = np.argsort(sims)[-args.top_k :][::-1]

    print("\n[Analogies]")
    print(f"   Original: {' + '.join(args.positive)} - {' - '.join(args.negative)}")
    print("--------------------------------------------------")
    print(f"   {'Word':<20} |   {'Cosine Similarity':<15}")
    print("--------------------------------------------------")
    for idx in top_indices:
        word = model.tokenizer.decode(idx)
        print(f"   {word:<20} |   {sims[idx]:.4f}")
    print("--------------------------------------------------\n")


if __name__ == "__main__":
    main()
