# src/data_preprocessing.py

import string
import os
from collections import Counter # Useful for word counts
from pickle import dump, load

# --- File Loading ---

def load_doc(filename):
    """Loads a text document into memory."""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except FileNotFoundError:
        print(f"Error: File not found at {filename}")
        return None

# --- Description Parsing and Cleaning ---

def load_descriptions(doc):
    """Parses descriptions from the Flickr8k token file into a dictionary."""
    if doc is None:
        return {}
    mapping = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        if len(line) < 2:
            continue
        image_id, image_desc = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        image_desc = ' '.join(image_desc)
        if image_id not in mapping:
            mapping[image_id] = list()
        mapping[image_id].append(image_desc)
    return mapping

def clean_descriptions(descriptions):
    """Cleans descriptions: lowercase, remove punctuation, remove short/numeric tokens."""
    if not descriptions:
        return
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            desc = desc.split()
            desc = [word.lower() for word in desc]
            desc = [w.translate(table) for w in desc]
            desc = [word for word in desc if len(word)>1]
            desc = [word for word in desc if word.isalpha()]
            desc_list[i] =  ' '.join(desc)

def add_start_end_tokens(descriptions):
    """Adds startseq and endseq tokens to each description."""
    updated_descriptions = {}
    if not descriptions:
        return updated_descriptions
    for key, desc_list in descriptions.items():
        updated_descriptions[key] = ['startseq ' + desc + ' endseq' for desc in desc_list]
    return updated_descriptions

# --- Vocabulary Creation ---

def to_lines(descriptions):
    """Converts dictionary of descriptions into a list of strings."""
    if not descriptions:
        return []
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

def create_vocabulary_mappings(descriptions, threshold=10):
    """Creates word vocabulary and word<->index mappings based on frequency."""
    if not descriptions:
        return set(), {}, {}, 0

    all_captions_text = to_lines(descriptions)
    # Count word occurrences
    word_counts = Counter()
    for sent in all_captions_text:
        word_counts.update(sent.split(' '))

    # Filter vocabulary based on threshold
    vocab = {w for w, count in word_counts.items() if count >= threshold}
    print(f'Vocabulary size (threshold={threshold}): {len(vocab)}')

    # Create mappings
    ixtoword = {}
    wordtoix = {}
    ix = 1 # Start index from 1 (0 usually reserved for padding)
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    # Vocabulary size for embedding layer (includes padding token 0)
    vocab_size = len(ixtoword) + 1
    print(f'Final vocab_size (incl. padding): {vocab_size}')

    return vocab, wordtoix, ixtoword, vocab_size

# --- Dataset ID Loading ---

def load_set(filename):
    """Loads a set of image identifiers from a file (train, test, dev)."""
    doc = load_doc(filename)
    if doc is None:
        return set()
    dataset = list()
    for line in doc.split('\n'):
        if len(line) < 1:
            continue
        identifier = line.split('.')[0] # Get ID part without .jpg
        dataset.append(identifier)
    return set(dataset)

# --- Utility for Max Length ---

def calculate_max_length(descriptions):
    """Calculates the maximum length of any description in the dictionary."""
    if not descriptions:
        return 0
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines) if lines else 0

# --- Prepare References for Evaluation ---

def prepare_references(descriptions, image_ids_set):
    """Prepares reference captions in tokenized format for evaluation."""
    references_tokenized = {}
    if not descriptions or not image_ids_set:
        return references_tokenized

    count_missing = 0
    for img_id in image_ids_set:
        if img_id in descriptions:
            tokenized_refs = [caption.split() for caption in descriptions[img_id]]
            # Use filename format if needed later, but ID is usually the key
            # filename_key = img_id + '.jpg'
            references_tokenized[img_id] = tokenized_refs
        else:
            count_missing += 1

    print(f"Prepared tokenized references for {len(references_tokenized)} image IDs.")
    if count_missing > 0:
        print(f"Warning: Missing descriptions for {count_missing} image IDs in reference set.")
    return references_tokenized

# --- Functions related to loading train/test/val descriptions ---
# Note: These now combine previous steps for clarity

def load_and_prepare_descriptions(token_path, image_ids_set):
    """Loads, cleans, and adds start/end tokens for a given set of image IDs."""
    doc = load_doc(token_path)
    all_descriptions = load_descriptions(doc)
    clean_descriptions(all_descriptions)

    # Filter descriptions for the specific dataset (train, val, or test)
    subset_descriptions = {
        img_id: captions
        for img_id, captions in all_descriptions.items()
        if img_id in image_ids_set
    }

    # Add start/end sequence tokens
    final_descriptions = add_start_end_tokens(subset_descriptions)

    return final_descriptions


def load_and_prepare_references(token_path, image_ids_set):
    """Loads, cleans, and tokenizes reference captions for evaluation."""
    doc = load_doc(token_path)
    all_descriptions = load_descriptions(doc)
    clean_descriptions(all_descriptions) # Clean originals

    # Prepare tokenized references directly
    references_tokenized = prepare_references(all_descriptions, image_ids_set)

    return references_tokenized