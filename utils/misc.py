import os
import re
import json
import ast
import random
import pandas as pd
from datetime import datetime, timedelta
import hashlib
import numpy as np
import torch

def get_text_hash(text):
    """
    Generate a unique ID (hash) for the given text.

    Args:
        text (str): The text to be hashed.

    Returns:
        str: The hashed ID.
    """
    hash_object = hashlib.sha256(text.encode())
    hashed_id = hash_object.hexdigest()
    return hashed_id

def check_list_formatting(text) -> str:
    """
    Check whether a given text is in an ordered list or an unordered list format.

    Args:
        text (_type_): _description_

    Returns:
        str: _description_
    """
    lines = text.strip().split('\n')
    for line in lines:
        if re.match(r'^\d+\.\s', line):
            # if re.match(r'^\d+\.\s.+:\s.+$', line):
            #     return "kv_ordered"
            # else:
            return "ordered"
        elif re.match(r'^[\*\-]\s', line):
            return "unordered"
        else:
            return "not_list"

def parse_kv_ordered_list(text) -> dict:
    """
    Args:
        text (_type_): 1. key: value\n2. key: value\n3. key: value\n

    Returns:
        dict: {1: {key: value}, 2: {key: value}, 3: {key: value}}
    """
    result_dict = {}
    lines = text.strip().split('\n')
    for line in lines:
        line = line.replace("*", "") # Remove all asterisks, if any. Models tend to output markdown bold text with asterisks.
        match = re.match(r'^(\d+)\.\s(.+)$', line.strip())
        if match:
            level = int(match.group(1))
            key_and_value = match.group(2)
            key = key_and_value.split(":")[0].strip()
            value = ":".join(key_and_value.split(":")[1:]).strip()
            result_dict[level] = {key: value}
    return result_dict

def capture_and_parse_ordered_list(text):
    # Match numbers followed by a period and a space
    ordered_list_pattern = re.compile(r'(\d+\.\s+.+?)(?=\d+\.\s|$)', re.DOTALL)
    matches = ordered_list_pattern.findall(text)
    
    # Remove numbers from the beginning of each match
    ordered_list = [re.sub(r'^\d+\.\s+', '', match).strip() for match in matches]
    
    if ordered_list:
        return ordered_list

    # Fallback: try literal lists and newline-separated values
    return parse_list_string(text)

def capture_and_parse_unordered_list(text):
    """
    Args:
        text (_type_): * value\n* value\n* value\n or - value\n- value\n- value\n
    
    Returns:
        list: [value1, value2, value3]
    """
    # Match asterisks or hyphens followed by a space
    unordered_list_pattern = re.compile(r'([\*\-]\s+.+?)(?=[\*\-]\s|$)', re.DOTALL)
    matches = unordered_list_pattern.findall(text)
    
    # Remove asterisks or hyphens from the beginning of each match
    unordered_list = [re.sub(r'^[\*\-]\s+', '', match).strip() for match in matches]

    if unordered_list:
        return unordered_list

    # Fallback: try literal lists and newline-separated values
    return parse_list_string(text)

def parse_list_string(text: str) -> list:
    """
    Parse a string that could be:
    - A JSON/Python list literal: ["a", "b"] or ['a', 'b']
    - Newline-separated values: a\nb\nc
    
    Args:
        text (str): The text to parse
    
    Returns:
        list: List of parsed items
    """
    stripped = text.strip()
    
    # Try bracket-enclosed list formats
    if stripped.startswith('[') and stripped.endswith(']'):
        try:
            # Try JSON first (handles ["item1", "item2"])
            result = json.loads(stripped)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
        try:
            # Fall back to Python literal (handles ['item1', 'item2'])
            result = ast.literal_eval(stripped)
            if isinstance(result, list):
                return result
        except (ValueError, SyntaxError):
            pass
    
    # Fall back to newline splitting
    lines = stripped.split("\n")
    return [line.strip() for line in lines if line.strip()]

def parse_ordered_list(text) -> list:
    """
    Args:
        text (_type_): 1. value\n2. value\n3. value\n

    Returns:
        list: [value1, value2, value3]
    """
    result_list = []
    lines = text.strip().split('\n')
    for line in lines:
        match = re.match(r'^\d+\.\s(.+)$', line.strip())
        if match:
            result_list.append(match.group(1))
    return result_list

def parse_unordered_list(text) -> list:
    """
    Args:
        text (_type_): * value\n* value\n* value\n or - value\n- value\n- value\n

    Returns:
        list: [value1, value2, value3]
    """
    result_list = []
    lines = text.strip().split('\n')
    for line in lines:
        match = re.match(r'^[\*\-]\s(.+)$', line.strip())
        if match:
            result_list.append(match.group(1))
    return result_list


def parse_multiple_choice(text) -> list:
    """
    Args:
        text (_type_): 3. value\n6. value\n7. value\n

    Returns:
        list: [3, 6, 7]
    """
    result_list = []
    lines = text.strip().split('\n')
    for line in lines:
        line = line.replace("*", "") # Remove all asterisks, if any. Models tend to output markdown bold text with asterisks.
        match = re.match(r'^(\d+)\.\s.+$', line.strip())
        if match:
            result_list.append(int(match.group(1)))
    return result_list

def parse_kv_multiple_choice(text) -> list:
    """
    Args:
        text (_type_): 3. key3: value\n6. key6: value\n7. key7: value\n

    Returns:
        list: [key3, key6, key7]
    """
    result_list = []
    lines = text.strip().split('\n')
    for line in lines:
        line = line.replace("*", "") # Remove all asterisks, if any. Models tend to output markdown bold text with asterisks.
        match = re.match(r'^(\d+)\.\s(.+)$', line.strip())
        if match:
            key_and_value = match.group(2)
            key = key_and_value.split(":")[0].strip()
            result_list.append(key)
    return result_list

def parse_markdown_list(text) -> list:
    list_format = check_list_formatting(text)
    if list_format == "ordered":
        return parse_ordered_list(text)
    elif list_format == "unordered":
        return parse_unordered_list(text)
    else:
        return None

def capture_speakers(chat):
    """
    Capture any string between the newline and the colon (:)
    """
    # re_pattern = r'(?<=\n)[a-zA-Z\s]*(?=:)'
    re_pattern = r'(?<=\n)(.*)(?=:)'
    speakers = re.findall(re_pattern, "\n" + chat)
    speakers = [s.strip() for s in speakers]

    return speakers

def capture_utterances(chat):
    re_pattern = r'(?<=:)(.*)(?=\n)'
    utterances = re.findall(re_pattern, chat)
    utterances = [u.strip() for u in utterances]
    return utterances

def clean_trailing_whitespaces(text):
    pattern = re.compile(r'\n\s+\n') # remove extra blank lines that contain only whitespace
    text = re.sub(pattern, '\n\n', text)

    pattern = re.compile(r'[^\S\r\n]+\n') # remove trailing whitespaces before newline
    text = re.sub(pattern, '\n', text)

    return text

def dump_jsonl(file_path: str, data: list, visual=False, suffix= ""):
    
    split = os.path.splitext(file_path)
    path, ext = split[0], split[1]
    file_path = f"{path}{suffix}{ext}"        

    with open(file_path, "a") as fp:
        for datum in data:
            fp.write(json.dumps(datum) + "\n")

    if visual:
        # Also write a version that is easy to view
        split = os.path.splitext(file_path)
        path, ext = split[0], split[1]
        file_path_visual = f"{path}_visual{ext}"
        with open(file_path_visual, "a") as fp:
            for datum in data:
                fp.write(json.dumps(datum, indent=4) + "\n")

def read_logs(file_path):
    if os.path.exists(file_path):
        logs = pd.read_json(file_path, lines=True)
        return logs
    else:
        return None

def generate_random_date(start_year, end_year):
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    time_between_dates = end_date - start_date
    random_number_of_days = random.randrange(time_between_dates.days)
    random_date = start_date + timedelta(days=random_number_of_days)
    return random_date.strftime("%Y-%m-%d")

def generate_random_age(min_age=19, max_age=85):
    # 19 is the minimum age for legal adulthood in most countries
    # the average life expectancy in the US is
    # Both sexes: 77.5 years
    # Males: 74.8 years
    # Females: 80.2 years
    # with standard deviation of 5 years
    return random.randint(min_age, max_age)

def generate_random_marital_status():
    return random.choice(["Married", "Widowed", "Separated", "Divorced", "Single"])

def generate_random_income_class():
    return random.choice(["Lower", "Lower-Middle", "Middle", "Upper-Middle", "Upper"])

def generate_random_from_list (list_items):
    return random.choice(list_items)

def generate_random_list_from_list (list_items,k):
    return random.sample(list_items,k)

def get_dict_basic_event_attrs(path='data/events-attrs/'):
    json_files =['basic_categories_attr_seed.jsonl','events_categories_attr_seed.jsonl','basic_attr_values_seed.jsonl','events_attr_values_seed.jsonl']
    dicts = []

    for file_name in json_files:
        temp_dict = {}
        file = open(path+'/'+file_name,'r')
        for line in file:
            line_dict = json.loads(line.strip()) 
            temp_dict = temp_dict | line_dict #merge what we loaded
        dicts.append(temp_dict)

    return {'basic':dicts[0],'events':dicts[1],'basic_vals':dicts[2],'event_vals':dicts[3]}

def compute_embedding_vendi(embeddings_matrix: np.ndarray, use_gpu: bool = True) -> float:
    """
    Computes the Embedding Vendi score from a matrix of embeddings.

    Parameters:
        embeddings_matrix (np.ndarray): Shape (n_samples, embedding_dim)
        use_gpu (bool): Whether to use GPU for computation (default: True)

    Returns:
        float: The exponentiated entropy of the eigenvalue distribution
    """
    # Input validation
    if not isinstance(embeddings_matrix, np.ndarray):
        raise TypeError("embeddings_matrix must be a numpy array")
    
    if embeddings_matrix.ndim != 2:
        raise ValueError("embeddings_matrix must be 2-dimensional")
    
    if embeddings_matrix.shape[0] == 0 or embeddings_matrix.shape[1] == 0:
        raise ValueError("embeddings_matrix cannot have empty dimensions")
    
    # Determine device based on matrix size and GPU availability
    matrix_size = embeddings_matrix.shape[0] * embeddings_matrix.shape[1]
    if use_gpu and torch.cuda.is_available() and matrix_size > 10000:  # Only use GPU for larger matrices
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        use_gpu = False  # Override to avoid unnecessary GPU operations

    with torch.no_grad():
        # Ensure the numpy array is contiguous and has the right dtype
        if not embeddings_matrix.flags['C_CONTIGUOUS']:
            embeddings_matrix = np.ascontiguousarray(embeddings_matrix)
        
        # Convert to float32 if needed (for numerical stability)
        if embeddings_matrix.dtype != np.float32:
            embeddings_matrix = embeddings_matrix.astype(np.float32)
        
        # Convert to PyTorch tensor efficiently
        if device.type == 'cpu':
            # For CPU, use from_numpy to avoid copying if possible
            embeddings_tensor = torch.from_numpy(embeddings_matrix)
        else:
            # For GPU, we need to copy anyway, so use torch.tensor
            embeddings_tensor = torch.tensor(embeddings_matrix, dtype=torch.float32, device=device)

        # Step 1: Normalize embeddings (important for stability)
        # Add small epsilon to avoid division by zero
        norms = torch.norm(embeddings_tensor, dim=1, keepdim=True)
        norms = torch.clamp(norms, min=1e-8)  # Prevent division by zero
        normalized = embeddings_tensor / norms

        # Step 2: Compute covariance matrix (dot product)
        K = torch.mm(normalized, normalized.T) / normalized.shape[0]

        # Step 3: Compute eigenvalues (only real part)
        eigenvalues = torch.linalg.eigvalsh(K)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]  # Filter out tiny or zero values to avoid log(0)

        # Check if we have any valid eigenvalues
        if len(eigenvalues) == 0:
            return 1.0  # Return 1.0 for degenerate case (all eigenvalues near zero)

        # Step 4: Normalize eigenvalues to form a probability distribution
        probs = eigenvalues / torch.sum(eigenvalues)

        # Step 5: Compute entropy of the distribution
        # Add small epsilon to avoid log(0)
        probs = torch.clamp(probs, min=1e-12)
        entropy = -torch.sum(probs * torch.log(probs))

        # Step 6: Exponentiate to get the Vendi score
        vendi_score = torch.exp(entropy)

    # Convert back to Python float
    return vendi_score.item()

def calculate_text_difference(text1: str, text2: str) -> float:
    """
    Calculate the difference between two text strings using character-level longest common substring.
    
    Args:
        text1 (str): First text string
        text2 (str): Second text string
        
    Returns:
        float: Difference between the two text strings
    """
    # Convert to lowercase and clean whitespace for comparison
    clean_text1 = text1.lower().strip()
    clean_text2 = text2.lower().strip()
    
    if not clean_text1 or not clean_text2:
        return 0.0
        
    from difflib import SequenceMatcher
    matcher = SequenceMatcher(None, clean_text1, clean_text2)
    lcs_length = matcher.find_longest_match(0, len(clean_text1), 0, len(clean_text2)).size
    
    # Normalize by the length of the longer text
    max_length = max(len(clean_text1), len(clean_text2))
    
    return max_length - lcs_length if max_length > 0 else 0.0