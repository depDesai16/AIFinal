import argparse
from generate import generate_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str, help="Input prompt")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--k", type=int, default=10, help="Top-k sampling size")
    parser.add_argument("--length", type=int, default=100, help="Number of characters to generate")
    args = parser.parse_args()

    # Convert prompt to lowercase to match training data preprocessing
    result = generate_text(args.prompt.lower(), args.length, temperature=args.temperature, k=args.k)
    print("Generated:", result)


if __name__ == "__main__":
    main()