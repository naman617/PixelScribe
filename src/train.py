# src/train.py

import os
import math
import pickle
import time
import argparse # For command-line arguments
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Import functions from other modules within the 'src' directory
# Use relative imports if running as part of a package,
# or ensure src is in PYTHONPATH if running directly
try:
    from .data_preprocessing import (load_doc, load_descriptions, clean_descriptions,
                                     create_vocabulary_mappings, load_set, calculate_max_length,
                                     load_and_prepare_descriptions, load_and_prepare_references)
    from .model import create_caption_model
    from .generator import data_generator
except ImportError:
    # Fallback for running script directly might require adjusting path
    # Or simply use absolute imports if src is added to PYTHONPATH
    print("Assuming running script directly, using absolute imports from src.")
    from data_preprocessing import (load_doc, load_descriptions, clean_descriptions,
                                     create_vocabulary_mappings, load_set, calculate_max_length,
                                     load_and_prepare_descriptions, load_and_prepare_references)
    from model import create_caption_model
    from generator import data_generator


def load_glove_embeddings(glove_path, embedding_dim, wordtoix, vocab_size):
    """Loads GloVe embeddings and creates an embedding matrix."""
    print(f"Loading GloVe embeddings from: {glove_path}")
    embeddings_index = {}
    try:
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
    except FileNotFoundError:
        print(f"ERROR: GloVe file not found at {glove_path}")
        return None
    print(f'Found {len(embeddings_index)} word vectors in GloVe file.')

    # Create embedding matrix
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    hits = 0
    misses = 0
    for word, i in wordtoix.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all zeros.
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print(f"Converted {hits} words ({misses} misses)")
    print(f"Embedding matrix shape: {embedding_matrix.shape}")
    return embedding_matrix


def main(args):
    """Main training function."""
    start_process_time = time.time()

    print("--- Loading Data IDs ---")
    train_ids = load_set(args.train_ids_path)
    dev_ids = load_set(args.dev_ids_path)
    print(f"Train IDs: {len(train_ids)}, Dev IDs: {len(dev_ids)}")

    # Basic check
    if not train_ids:
        print("ERROR: Training IDs could not be loaded. Exiting.")
        return

    use_validation = bool(dev_ids)
    if not use_validation:
         print("WARNING: Dev IDs not loaded. Proceeding without validation.")

    print("\n--- Loading and Preparing Descriptions ---")
    train_descriptions = load_and_prepare_descriptions(args.token_path, train_ids)
    val_descriptions = {}
    if use_validation:
        val_descriptions = load_and_prepare_descriptions(args.token_path, dev_ids)
    print(f"Train Descriptions: {len(train_descriptions)}")
    if use_validation: print(f"Validation Descriptions: {len(val_descriptions)}")

    if not train_descriptions:
        print("ERROR: Training descriptions could not be prepared. Exiting.")
        return

    print("\n--- Creating Vocabulary ---")
    # Create vocab from TRAINING descriptions only (before start/end tokens added is fine)
    # To do this cleanly, we reload and clean the base descriptions
    doc = load_doc(args.token_path)
    all_descriptions_base = load_descriptions(doc)
    clean_descriptions(all_descriptions_base)
    train_desc_base = {k: v for k, v in all_descriptions_base.items() if k in train_ids}

    vocab, wordtoix, ixtoword, vocab_size = create_vocabulary_mappings(
        train_desc_base, threshold=args.vocab_threshold
    )
    # Save vocab mappings (optional but recommended)
    mappings_path = os.path.join(args.output_dir, 'vocab_mappings.pkl')
    with open(mappings_path, 'wb') as f:
        pickle.dump({'wordtoix': wordtoix, 'ixtoword': ixtoword, 'vocab_size': vocab_size}, f)
    print(f"Vocabulary mappings saved to {mappings_path}")


    print("\n--- Calculating Max Length ---")
    # Calculate max length based on the prepared training descriptions (with start/end tokens)
    max_length = calculate_max_length(train_descriptions)
    print(f"Max Description Length: {max_length}")
    # Save max_length (optional)
    # max_length_path = os.path.join(args.output_dir, 'max_length.pkl')
    # with open(max_length_path, 'wb') as f: pickle.dump(max_length, f)


    print("\n--- Loading Image Features ---")
    try:
        with open(args.train_features_path, 'rb') as f:
            train_features = pickle.load(f)
        print(f"Loaded {len(train_features)} training features.")
    except FileNotFoundError:
         print(f"ERROR: Training features not found at {args.train_features_path}. Exiting.")
         return
    except Exception as e:
         print(f"Error loading training features: {e}. Exiting.")
         return

    val_features = {}
    if use_validation:
        # For validation features, need to potentially load train+test and filter
        print("Loading features for validation set...")
        all_loaded_features = {}
        if os.path.exists(args.train_features_path):
            all_loaded_features.update(load(open(args.train_features_path, "rb")))
        if args.test_features_path and os.path.exists(args.test_features_path):
             all_loaded_features.update(load(open(args.test_features_path, "rb")))
        elif not all_loaded_features:
             print("WARNING: Cannot create validation features - no feature files loaded.")
             use_validation = False # Disable validation if features missing
        else:
             val_features = {img_id_jpg: feature
                             for img_id_jpg, feature in all_loaded_features.items()
                             if img_id_jpg.split('.')[0] in dev_ids}
             print(f"Prepared {len(val_features)} validation features.")
             del all_loaded_features # Free memory

    print("\n--- Loading GloVe Embeddings ---")
    embedding_matrix = load_glove_embeddings(args.glove_path, args.embedding_dim, wordtoix, vocab_size)
    if embedding_matrix is None:
        print("ERROR: Failed to load GloVe embeddings. Exiting.")
        return

    print("\n--- Defining Model ---")
    model = create_caption_model(vocab_size, max_length, args.embedding_dim,
                                 feature_dim=2048, lstm_units=args.lstm_units) # Assuming 2048 features

    print("\n--- Setting Embedding Layer Weights ---")
    # Find the embedding layer (assuming it's the third layer, index 2 - CHECK model.summary() output!)
    try:
        embedding_layer_index = -1
        for i, layer in enumerate(model.layers):
            if isinstance(layer, tf.keras.layers.Embedding):
                embedding_layer_index = i
                break
        if embedding_layer_index != -1:
            print(f"Setting weights for Embedding layer (index {embedding_layer_index})")
            model.layers[embedding_layer_index].set_weights([embedding_matrix])
            model.layers[embedding_layer_index].trainable = False # Freeze embedding layer
        else:
            print("WARNING: Could not find Embedding layer to set weights.")
    except Exception as e:
        print(f"Error setting embedding weights: {e}")


    print("\n--- Compiling Model ---")
    # Using Adam optimizer with default learning rate initially
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary() # Print summary after compiling

    print("\n--- Preparing Training ---")
    # Calculate steps
    steps_per_epoch = math.ceil(len(train_descriptions) / args.batch_size)
    validation_steps = 0
    if use_validation:
        validation_steps = math.ceil(len(val_descriptions) / args.batch_size)
    print(f"Training steps per epoch: {steps_per_epoch}")
    if use_validation: print(f"Validation steps: {validation_steps}")

    # Define Callbacks
    callbacks_list = []
    weights_save_path = os.path.join(args.output_dir, 'best_model.weights.h5')
    if use_validation:
        checkpoint = ModelCheckpoint(filepath=weights_save_path, monitor='val_loss', verbose=1,
                                     save_best_only=True, save_weights_only=True, mode='min')
        callbacks_list.append(checkpoint)

        early_stopping = EarlyStopping(monitor='val_loss', patience=args.patience, verbose=1,
                                       mode='min', restore_best_weights=True)
        callbacks_list.append(early_stopping)
        print(f"Callbacks enabled: ModelCheckpoint (best val_loss -> {weights_save_path}), EarlyStopping (patience={args.patience})")
    else:
        # Save weights at the end if no validation
        print("Callbacks disabled (no validation data). Will save final weights at end.")


    # Create Generators
    print("Creating data generators...")
    # Remember to pass vocab_size to the generator!
    train_generator = data_generator(train_descriptions, train_features, wordtoix, max_length, args.batch_size, vocab_size)
    validation_generator = None
    if use_validation:
        validation_generator = data_generator(val_descriptions, val_features, wordtoix, max_length, args.batch_size, vocab_size)

    # --- Start Training ---
    print(f"\n--- Starting Training (Max Epochs: {args.epochs}, Batch Size: {args.batch_size}) ---")
    training_start_time = time.time()

    history = model.fit(
        train_generator,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps if use_validation else None,
        callbacks=callbacks_list if callbacks_list else None,
        verbose=1
    )

    training_end_time = time.time()
    print("\n--- Training Complete ---")
    print(f"Total training time: {training_end_time - training_start_time:.2f} seconds")

    # --- Outcome & Final Save ---
    if use_validation and early_stopping.stopped_epoch > 0:
        print(f"Training stopped early at epoch {early_stopping.stopped_epoch + 1}.")
        print(f"Best weights (restored) correspond to val_loss: {early_stopping.best:.4f}")
        print(f"Best weights also saved to: {weights_save_path}")
    elif use_validation:
         print(f"Training completed {args.epochs} epochs.")
         print(f"Best weights saved by ModelCheckpoint to: {weights_save_path}")
    else:
        # No validation - save the final weights manually
        final_weights_path = os.path.join(args.output_dir, f'final_model_epoch_{args.epochs}.weights.h5')
        model.save_weights(final_weights_path)
        print(f"No validation performed. Final model weights saved to: {final_weights_path}")

    print(f"\nTotal script time: {time.time() - start_process_time:.2f} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an Image Captioning Model')

    # File Paths
    parser.add_argument('--token_path', type=str, required=True, help='Path to Flickr8k.token.txt')
    parser.add_argument('--train_ids_path', type=str, required=True, help='Path to Flickr_8k.trainImages.txt')
    parser.add_argument('--dev_ids_path', type=str, default=None, help='Path to Flickr_8k.devImages.txt (optional, for validation)')
    parser.add_argument('--train_features_path', type=str, required=True, help='Path to encoded_train_images.pkl')
    parser.add_argument('--test_features_path', type=str, default=None, help='Path to encoded_test_images.pkl (needed if using validation)')
    parser.add_argument('--glove_path', type=str, required=True, help='Path to GloVe embedding file (e.g., glove.6B.200d.txt)')
    parser.add_argument('--output_dir', type=str, default='./training_output', help='Directory to save model weights and vocab mappings')

    # Model Hyperparameters
    parser.add_argument('--embedding_dim', type=int, default=200, help='Dimension of GloVe embeddings')
    parser.add_argument('--lstm_units', type=int, default=256, help='Number of units in LSTM layer')
    parser.add_argument('--vocab_threshold', type=int, default=10, help='Minimum word frequency for vocabulary inclusion')

    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=50, help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=3, help='Batch size (number of photos per generator yield)')
    parser.add_argument('--patience', type=int, default=5, help='Patience for EarlyStopping')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)