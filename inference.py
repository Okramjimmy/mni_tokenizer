"""
Inference script for Meitei Mayek Sentence Splitter.

This script demonstrates how to use the trained model to split
Meitei Mayek text into sentences with context awareness.
"""

import argparse
import spacy
from meitei_tokenizer import MeiteiTokenizer


def load_model(model_path: str, sentencepiece_path: str) -> spacy.Language:
    """
    Load the trained spaCy model with the custom tokenizer.

    Args:
        model_path: Path to the trained spaCy model directory.
        sentencepiece_path: Path to the SentencePiece .model file.

    Returns:
        A spaCy Language object ready for inference.
    """
    nlp = spacy.load(model_path)
    # Replace the default tokenizer with our custom one
    nlp.tokenizer = MeiteiTokenizer(sentencepiece_path, nlp.vocab)
    return nlp


def split_sentences(nlp: spacy.Language, text: str) -> list:
    """
    Split text into sentences using the trained model.

    Args:
        nlp: The spaCy Language object.
        text: Input text to split.

    Returns:
        List of sentence strings.
    """
    doc = nlp(text)
    return [sent.text for sent in doc.sents]


def main():
    parser = argparse.ArgumentParser(
        description="Split Meitei Mayek text into sentences"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="./output/model-best",
        help="Path to trained spaCy model",
    )
    parser.add_argument(
        "--sentencepiece",
        "-s",
        type=str,
        default="meitei_tokenizer.model",
        help="Path to SentencePiece .model file",
    )
    parser.add_argument(
        "--text",
        "-t",
        type=str,
        help="Text to split into sentences",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode",
    )

    args = parser.parse_args()

    print("Loading model...")
    nlp = load_model(args.model, args.sentencepiece)
    print("Model loaded.")

    if args.text:
        sentences = split_sentences(nlp, args.text)
        print("\nSentences:")
        for i, sent in enumerate(sentences, 1):
            print(f"  {i}. {sent}")

    if args.interactive:
        print("\nInteractive mode. Type 'quit' to exit.\n")
        while True:
            text = input("Enter text: ").strip()
            if text.lower() == "quit":
                break
            if not text:
                continue

            sentences = split_sentences(nlp, text)
            print(f"\nFound {len(sentences)} sentence(s):")
            for i, sent in enumerate(sentences, 1):
                print(f"  {i}. {sent}")
            print()


if __name__ == "__main__":
    main()
