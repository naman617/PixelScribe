# src/evaluate.py

import os
import pickle
import random
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import nltk
# Ensure NLTK data is available (run once if needed: nltk.download('punkt'))
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt', quiet=True)
    print("'punkt' tokenizer downloaded.")
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Import necessary functions/classes from other src modules
# Assuming you run this script from the main PixelScribe directory
# or have added 'src' to your PYTHONPATH
try:
    from src.predict import greedy_search_predict # Or beam_search_decoder if implemented
    from src.model import create_caption_model
    from src.data_preprocessing import load_and_prepare_references, load_set
except ImportError as e:
    print(f"Error importing modules from src: {e}")
    print("Ensure you are running this script from the project root directory,")
    print("or that the 'src' directory is in your PYTHONPATH.")
    # Fallback for simpler execution if files are in the same directory (less ideal)
    try:
         from predict import greedy_search_predict
         from model import create_caption_model
         from data_preprocessing import load_and_prepare_references, load_set
         print("Using direct imports assuming scripts are in the same directory.")
    except ImportError:
         print("Failed to import necessary functions. Please check project structure and PYTHONPATH.")
         exit()


def main(args):
    """Loads model, evaluates captions on random examples using sentence BLEU."""

    print("--- Loading Vocabulary Mappings & Max Length ---")
    try:
        with open(args.vocab_path, 'rb') as f:
            mappings = pickle.load(f)
            wordtoix = mappings['wordtoix']
            ixtoword = mappings['ixtoword']
            vocab_size = mappings['vocab_size']
        with open(args.max_length_path, 'rb') as f:
            max_length = pickle.load(f)
        print(f"Loaded vocab (size={vocab_size}), mappings, and max_length ({max_length}).")
        if not isinstance(max_length, int) or max_length <= 0:
             raise ValueError(f"Loaded max_length is invalid: {max_length}")
    except FileNotFoundError:
        print(f"ERROR: Vocab or max_length file not found. Check paths:")
        print(f"  Vocab path: {args.vocab_path}")
        print(f"  Max length path: {args.max_length_path}")
        return
    except Exception as e:
        print(f"Error loading vocab/max_length: {e}")
        return

    print("\n--- Defining Captioning Model ---")
    # Ensure model parameters match the trained model
    caption_model = create_caption_model(vocab_size, max_length, args.embedding_dim,
                                         args.feature_dim, args.lstm_units)

    print("\n--- Loading Trained Weights ---")
    if not os.path.exists(args.weights_path):
        print(f"ERROR: Model weights file not found at {args.weights_path}")
        return
    try:
        caption_model.load_weights(args.weights_path)
        print(f"Loaded weights from {args.weights_path}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    print("\n--- Loading Test Features ---")
    try:
        with open(args.features_path, "rb") as f:
            encoding_test = pickle.load(f)
        print(f"Successfully loaded {len(encoding_test)} test encodings.")
    except FileNotFoundError:
        print(f"ERROR: Test features file not found at {args.features_path}")
        return
    except Exception as e:
        print(f"An error occurred loading the test features pickle file: {e}")
        return

    print("\n--- Loading and Preparing Reference Captions ---")
    test_ids = load_set(args.test_ids_path)
    if not test_ids:
        print(f"ERROR: Could not load test IDs from {args.test_ids_path}")
        return
    # This function loads original tokens, cleans, filters by test_ids, and tokenizes refs
    # Assumes load_and_prepare_references is imported correctly
    test_references_tokenized = load_and_prepare_references(args.token_path, test_ids)
    if not test_references_tokenized:
        print("ERROR: Failed to prepare reference captions.")
        return

    # --- Select images to evaluate ---
    print(f"\n--- Evaluating {args.num_examples} Random Examples ---")
    # Ensure we only select images that have both features and references
    available_feature_keys = list(encoding_test.keys())
    available_ref_ids = list(test_references_tokenized.keys())
    # Find IDs present in both features (key format 'id.jpg') and references (key format 'id')
    available_ids_intersect = [k.split('.')[0] for k in available_feature_keys if k.split('.')[0] in available_ref_ids]

    if len(available_ids_intersect) < args.num_examples:
        print(f"Warning: Only {len(available_ids_intersect)} images available with both features and references.")
        num_examples_to_run = len(available_ids_intersect)
    else:
        num_examples_to_run = args.num_examples

    if num_examples_to_run <= 0:
        print("Cannot evaluate examples - no images found with both features and references.")
        return

    # Select random image IDs (without .jpg)
    example_ids = random.sample(available_ids_intersect, num_examples_to_run)

    # --- Initialize Smoothing Function for BLEU ---
    chencherry = SmoothingFunction()
    all_results = []

    # --- Loop through examples ---
    for image_id in example_ids:
        image_filename = image_id + '.jpg' # Reconstruct filename for features/display

        # --- Prepare Data ---
        if image_filename not in encoding_test: continue # Should not happen now
        if image_id not in test_references_tokenized: continue # Should not happen now

        photo_encoding = encoding_test[image_filename].reshape((1, args.feature_dim))
        references = test_references_tokenized[image_id] # List of tokenized reference lists

        # --- Generate Caption (Using Greedy Search imported from predict.py) ---
        try:
            # Make sure greedy_search_predict function is imported
            # It needs the model, encoding, wordtoix, ixtoword, max_length
            generated_caption_str = greedy_search_predict(caption_model, photo_encoding, wordtoix, ixtoword, max_length)
            generated_caption_tokens = generated_caption_str.split()
        except Exception as e:
            print(f"\nERROR generating caption for {image_filename}: {e}")
            generated_caption_str = "Generation Error"
            generated_caption_tokens = []


        # --- Calculate BLEU Scores ---
        bleu_scores = {}
        scores_str = "N/A"
        if generated_caption_tokens: # Only calculate if caption was generated
            try:
                bleu_scores['BLEU-1'] = sentence_bleu(references, generated_caption_tokens, weights=(1, 0, 0, 0), smoothing_function=chencherry.method1)
                bleu_scores['BLEU-2'] = sentence_bleu(references, generated_caption_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=chencherry.method1)
                bleu_scores['BLEU-3'] = sentence_bleu(references, generated_caption_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=chencherry.method1)
                bleu_scores['BLEU-4'] = sentence_bleu(references, generated_caption_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)
                scores_str = ", ".join([f"{key}={score:.3f}" for key, score in bleu_scores.items()])
            except Exception as e:
                 print(f"\nERROR calculating BLEU for {image_filename}: {e}")
                 bleu_scores = None
        else:
            bleu_scores = None

        # --- Store and Print Results ---
        result = {
            'image_id': image_id,
            'filename': image_filename,
            'generated_caption': generated_caption_str,
            'reference_example': ' '.join(references[0]) if references else "N/A", # Show first reference
            'bleu_scores_str': scores_str
        }
        all_results.append(result)

        print(f"\n--- Image: {image_filename} ---")
        # Display Image (Optional)
        if not args.no_display:
             try:
                 image_display_path = os.path.join(args.image_dir, image_filename) # Need image dir path
                 if os.path.exists(image_display_path):
                     img_display = Image.open(image_display_path)
                     plt.imshow(img_display)
                     plt.axis('off')
                     plt.show() # plt.show() might not work well outside notebooks
                 else:
                     print(f"(Could not display image: Path not found at {image_display_path})")
             except Exception as e:
                 print(f"(Error displaying image: {e})")

        # Print Captions and Scores
        print(f"  Generated Caption: {result['generated_caption']}")
        print(f"  Reference Example: {result['reference_example']}")
        print(f"  BLEU Scores: {result['bleu_scores_str']}")
        print("------------------------------------")

    # --- Optionally save results to a file ---
    if args.output_file:
        try:
            with open(args.output_file, 'w') as f:
                for res in all_results:
                    f.write(f"Image: {res['filename']}\n")
                    f.write(f"Generated: {res['generated_caption']}\n")
                    f.write(f"Reference: {res['reference_example']}\n")
                    f.write(f"BLEU Scores: {res['bleu_scores_str']}\n")
                    f.write("------------------------------------\n")
            print(f"\nEvaluation results saved to {args.output_file}")
        except Exception as e:
            print(f"\nError saving evaluation results to file: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Image Captioning Model on Examples with BLEU scores.')

    # Required Paths
    parser.add_argument('--weights_path', type=str, required=True, help='Path to trained model weights (.weights.h5).')
    parser.add_argument('--features_path', type=str, required=True, help='Path to encoded test features (.pkl).')
    parser.add_argument('--token_path', type=str, required=True, help='Path to the original Flickr8k.token.txt file.')
    parser.add_argument('--test_ids_path', type=str, required=True, help='Path to the Flickr_8k.testImages.txt file.')
    parser.add_argument('--vocab_path', type=str, required=True, help='Path to the vocabulary mappings pickle file (saved by train.py).')
    parser.add_argument('--max_length_path', type=str, required=True, help='Path to the max length pickle file (saved by train.py).')
    parser.add_argument('--image_dir', type=str, required=True, help='Path to the directory containing original test JPG images (for display).')


    # Optional Model Hyperparameters (must match trained model)
    parser.add_argument('--embedding_dim', type=int, default=200, help='Dimension of word embeddings.')
    parser.add_argument('--lstm_units', type=int, default=256, help='Number of units in LSTM layer.')
    parser.add_argument('--feature_dim', type=int, default=2048, help='Dimension of image features.')

    # Evaluation Parameters
    parser.add_argument('--num_examples', type=int, default=5, help='Number of random test examples to evaluate.')
    parser.add_argument('--output_file', type=str, default=None, help='Optional path to save evaluation results text file.')
    parser.add_argument('--no_display', action='store_true', help='Flag to suppress displaying images.')
    # Add argument if using beam search:
    # parser.add_argument('--use_beam', action='store_true', help='Flag to use Beam Search instead of Greedy.')
    # parser.add_argument('--beam_width', type=int, default=3, help='Beam width if using Beam Search.')


    args = parser.parse_args()

    main(args)