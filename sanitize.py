import os
import re
import json
import time
import random
import argparse
import copy
import threading
import logging
import functools
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from difflib import get_close_matches
from copy import deepcopy
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
from rouge import Rouge

from tqdm import tqdm
import colorful as cf
cf.use_true_colors()
cf.use_style('monokai')

from rich import print
from rich.panel import Panel
from rich import box

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress noisy third-party library logs
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ============== Custom Exceptions ==============
class SanitizationError(Exception):
    """Base exception for Privasis sanitization errors."""
    pass


class AttributeSelectionError(SanitizationError):
    """Error during attribute selection for sanitization."""
    pass


class SpanExtractionError(SanitizationError):
    """Error during span extraction from record."""
    pass


class SequenceSanitizationError(SanitizationError):
    """Error during sequence sanitization."""
    pass


class RecordRebuildError(SanitizationError):
    """Error during record reconstruction."""
    pass


class ModelInteractionError(SanitizationError):
    """Error during model interaction."""
    pass


class InstructionGenerationError(SanitizationError):
    """Error during instruction generation."""
    pass


# ============== Retry Decorator ==============
def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0, 
          exceptions: tuple = (Exception,), on_retry: callable = None):
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry
        on_retry: Optional callback function called on each retry with (attempt, exception)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        if on_retry:
                            on_retry(attempt + 1, e)
                        else:
                            logger.warning(f"Attempt {attempt + 1}/{max_attempts} for {func.__name__} failed: {e}")
                        time.sleep(current_delay)
                        current_delay *= backoff
            raise last_exception
        return wrapper
    return decorator


from model_client import load_model
from utils.load_data import load_privasis
from utils.misc import (
    read_logs,
    capture_and_parse_ordered_list,
    parse_list_string,
    clean_trailing_whitespaces,
    calculate_text_difference
)

PROJECT_HOME = Path(__file__).parent.resolve()
DATA_DIR = os.path.join(PROJECT_HOME, 'data')
OUTPUT_DIR = os.path.join(PROJECT_HOME, 'outputs', 'sanitized_privasis')

# set random seed
random.seed(42)

class PrivasisSanitizer():
    """
    Sanitize the Privasis data
    run() method is the main method to run the sanitization process, which is done by the sanitize_privasis_data() method
    """
    def __init__(self, args, online=False) -> None:
        self.args = args
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        if online:
            self.output_file = os.path.join(OUTPUT_DIR, f"{args.run_id}_sanitized.jsonl")
            self.output_file_visual = os.path.join(OUTPUT_DIR, f"{args.run_id}_sanitized_visual.jsonl")
            self.error_file = os.path.join(OUTPUT_DIR, f"{args.run_id}_sanitization_errors.jsonl")
        else:
            self.output_file = os.path.join(OUTPUT_DIR, f"{args.privasis_data_id}_{args.run_id}.jsonl")
            self.output_file_visual = os.path.join(OUTPUT_DIR, f"{args.privasis_data_id}_{args.run_id}_visual.jsonl")
            self.error_file = os.path.join(OUTPUT_DIR, f"{args.privasis_data_id}_{args.run_id}_errors.jsonl")
        self.retry_limit = args.retry_limit
        self.online = online
        
        # vLLM server configuration (for parallel processing)
        self.vllm_server_url = getattr(args, 'vllm_server_url', None)
        self.num_workers = getattr(args, 'num_workers', 1)
        
        # Thread-safe lock for file operations
        self._file_lock = threading.Lock()
        
        # Pre-load the model once at initialization
        self.model_name = args.sanitization_model
        self.model = load_model(self.model_name, vllm_server_url=self.vllm_server_url)

    def _replace_periods_with_markers(self, text: str) -> str:
        """
        Replace periods in various contexts with temporary markers to prevent incorrect sentence splitting.
        """
        title_patterns = [
            # URLs and domains
            (r'https?://[^\s]+', lambda m: m.group(0).replace('.', '_URL_DOT_')),
            (r'www\.[^\s]+', lambda m: m.group(0).replace('.', '_URL_DOT_')),
            (r'[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?', lambda m: m.group(0).replace('.', '_URL_DOT_')),
            # Floating point numbers
            (r'\b\d+\.\d+\b', lambda m: m.group(0).replace('.', '_FLOAT_DOT_')),
            # Version numbers
            (r'\b\d+\.\d+(?:\.\d+)*\b', lambda m: m.group(0).replace('.', '_VERSION_DOT_')),
            # Reference codes or identifiers (like U.2020.432V)
            (r'\b[A-Z]\.\d{4}\.[A-Z0-9]+\b', lambda m: m.group(0).replace('.', '_REF_DOT_')),
            # Country and organization abbreviations (both capitalized and lowercase)
            (r'\b([Uu]\.?[Ss]\.?[Aa]\.?|[Uu]\.?[Kk]\.?|[Uu]\.?[Nn]\.?|[Ee]\.?[Uu]\.?|[Nn]\.?[Aa]\.?[Tt]\.?[Oo]\.?|[Cc]\.?[Ii]\.?[Aa]\.?|[Ff]\.?[Bb]\.?[Ii]\.?|[Cc]\.?[Ee]\.?[Oo]\.?|[Cc]\.?[Ff]\.?[Oo]\.?|[Cc]\.?[Tt]\.?[Oo]\.?|[Dd]\.?[Pp]\.?|[Ee]\.?[Tt]\.?[Aa]\.?|[Gg]\.?[Mm]\.?[Tt]\.?|[Ii]\.?[Dd]\.?|[Ii]\.?[Pp]\.?|[Mm]\.?[Aa]\.?[Pp]\.?|[Nn]\.?[Aa]\.?[Ss]\.?[Aa]\.?|[Pp]\.?[Oo]\.?[Bb]\.?[Oo]\.?|[Pp]\.?[Ss]\.?[Ii]\.?|[Rr]\.?[Ss]\.?[Vp]\.?|[Ss]\.?[E]\.?[Oo]\.?|[Ss]\.?[Mm]\.?[Ss]\.?|[Ss]\.?[Oo]\.?[Ss]\.?|[Ss]\.?[Ss]\.?[Nn]\.?|[Tt]\.?[Cc]\.?[Pp]\.?|[Uu]\.?[Dp]\.?[Cc]\.?|[Uu]\.?[Rr]\.?[Ll]\.?|[Ww]\.?[Ww]\.?[Ww]\.?)\b', lambda m: m.group(0).replace('.', '_ORG_DOT_')),
            # Common titles (both capitalized and lowercase)
            (r'\b([Mm]r|[Mm]rs|[Dd]r|[Pp]rof|[Rr]ev|[Ss]r|[Jj]r|[Ss]t|[Mm]s|[Cc]apt|[Cc]ol|[Gg]en|[Ll]t|[Ss]gt|[Cc]pl|[Pp]vt|[Mm]aj|[Cc]mdr|[Aa]dm|[Gg]ov|[Ss]en|[Rr]ep|[Pp]res|[Vv]ice [Pp]res)\.', lambda m: m.group(1) + '_TITLE_DOT_'),
            # Common abbreviations (both capitalized and lowercase)
            (r'\b([Ee]\.g\.|[Ii]\.e\.|[Ee]tc\.|[Vv]s\.|[Aa]pprox\.|[Ee]st\.|[Dd]ept\.|[Ff]ig\.|[Nn]o\.|[Vv]ol\.|[Pp]\.|[Pp]p\.|[Cc]h\.|[Ss]ec\.|[Rr]ef\.|[Ee]x\.|[Ii]nc\.|[Ll]td\.|[Cc]o\.|[Cc]orp\.|[Aa]ve\.|[Ss]t\.|[Rr]d\.|[Bb]lvd\.)', lambda m: m.group(1).replace('.', '_ABBR_DOT_'))
        ]
        
        temp_text = text
        for pattern, replacement in title_patterns:
            temp_text = re.sub(pattern, replacement, temp_text)
        return temp_text

    def _restore_periods_from_markers(self, text: str) -> str:
        """
        Restore periods from temporary markers back to their original form.
        """
        return (text
            .replace('_URL_DOT_', '.')
            .replace('_WWW_DOT_', '.')
            .replace('_DOMAIN_DOT_', '.')
            .replace('_FLOAT_DOT_', '.')
            .replace('_VERSION_DOT_', '.')
            .replace('_REF_DOT_', '.')
            .replace('_ORG_DOT_', '.')
            .replace('_TITLE_DOT_', '.')
            .replace('_ABBR_DOT_', '.'))

    def _split_chunk_by_newline(self, text: str, max_chunk_length: int) -> list:
        """
        Recursively split the chunk into sub-chunks according to new lines until the chunk is shorter than max_chunk_length
        """
        if len(text) < max_chunk_length:
            return [text]
        elif len(text.split("\n")) == 1:
            return self._split_chunk_by_sentence(text, max_chunk_length)
        else:
            subseqs = text.split("\n")
            mid = len(subseqs) // 2
            left_seq = "\n".join(subseqs[:mid])
            right_seq = "\n".join(subseqs[mid:])
            return self._split_chunk_by_newline(left_seq, max_chunk_length) + self._split_chunk_by_newline(right_seq, max_chunk_length)

    def _split_chunk_by_sentence(self, text: str, max_chunk_length: int) -> list:
        """
        Recursively split the chunk into sub-chunks according to sentences until the chunk is shorter than max_chunk_length
        """
        if len(text) < max_chunk_length or len(re.split(r'\.\s+', text)) == 1:
            return [text]
        else:
            subseqs = re.split(r'\.\s+', text)
            mid = len(subseqs) // 2
            left_seq = ". ".join(subseqs[:mid]) # BUG: if there is a period followed by a new line, the new line will be removed during the split above and here we will be replacing the newline with a space!
            right_seq = ". ".join(subseqs[mid:])
            return self._split_chunk_by_sentence(left_seq, max_chunk_length) + self._split_chunk_by_sentence(right_seq, max_chunk_length)

    def decompose_record(self, record: dict) -> dict:
        """
        Recursively split the record into sequences until the chunk is shorter than max_chunk_length
        """
        text = clean_trailing_whitespaces(record['text']) # clean up record

        sequences = []
        seq_idx = 0
        seqs = text.split("\n\n")

        max_chunk_length = 512

        for seq in seqs:
            if len(seq) < max_chunk_length:
                sequences.append({'seq': seq, 'terminator': "\n\n", 'idx': seq_idx})
                seq_idx += 1
            else: # if it's too long, recursively split the chunk in half until the chunk is shorter than max_chunk_length
                temp_seq = self._replace_periods_with_markers(seq)

                num_lines = len(temp_seq.split("\n")) # get number of lines in the sequence

                if num_lines > 1:
                    # recursively split the chunk in half according to new lines until the chunk is shorter than max_chunk_length
                    subseqs = self._split_chunk_by_newline(temp_seq, max_chunk_length)
                    for subseq in subseqs[:-1]:
                        restored_subseq = self._restore_periods_from_markers(subseq)
                        sequences.append({'seq': restored_subseq, 'terminator': "\n", 'idx': seq_idx})
                        seq_idx += 1
                    restored_subseq = self._restore_periods_from_markers(subseqs[-1])
                    sequences.append({'seq': restored_subseq, 'terminator': "\n\n", 'idx': seq_idx})
                    seq_idx += 1
                else: # if there are no new lines, then split on sentences
                    subseqs = self._split_chunk_by_sentence(temp_seq, max_chunk_length)
                    for subseq in subseqs[:-1]:
                        restored_subseq = self._restore_periods_from_markers(subseq)
                        sequences.append({'seq': restored_subseq + ".", 'terminator': " ", 'idx': seq_idx})
                        seq_idx += 1
                    restored_subseq = self._restore_periods_from_markers(subseqs[-1])
                    sequences.append({'seq': restored_subseq, 'terminator': "\n\n", 'idx': seq_idx})
                    seq_idx += 1

        return {'sequences': sequences, 'text': text}

    def rebuild_record(self, sequences: list) -> dict:
        """
        Rebuild the record from the sequences
        """
        text = ""
        for seq in sequences:
            text += seq['seq'] + seq['terminator']
        return text.strip()

    def attribute_weighting(self, attributes: dict, is_group: bool = False) -> dict:
        if self.args.attr_selection_weighting == "uniform":
            return {key: 1 for key in attributes.keys()}
        elif self.args.attr_selection_weighting == "sensitivity":
            if is_group:
                prompt = f"{json.dumps(attributes, indent=4)}\n\nGiven the group of attributes, rank the sensitivity. Subjective attributes are less sensitive than objective attributes. Sort the group names in an ordered list without any additional comments or explanations."
            else:
                prompt = f"{json.dumps(attributes, indent=4)}\n\nGiven the values of the attributes, rank the sensitivity. Subjective attributes are less sensitive than objective attributes. Sort the attributes in an ordered list without any additional comments or explanations."
            model = self.model
            results = model.interact(prompt, temperature=0.7)
            ordered_keys = capture_and_parse_ordered_list(results)
            if ordered_keys == []:
                # Handle both bracket-enclosed lists and newline-separated values
                ordered_keys = parse_list_string(results)
            # get weights based on the ranking. the first one is the most sensitive with the highest weight 1.3 and the last one is the least sensitive with the lowest weight 0.3
            weights = {key: 0.3 + 1/(idx+1) for idx, key in enumerate(ordered_keys)}
            weights = {key: val/sum(weights.values()) for key, val in weights.items()}
            return weights
        else:
            raise NotImplementedError("The attribute selection weighting should be either 'uniform' or 'sensitivity'.")

    def select_target_attrbutes(self, grouped_attrs, individual_attrs, n=None, group_individual_ratio=[8, 2], weight_inverse=False) -> dict:
        target_granularity = random.choices(['group', 'individual'], weights=group_individual_ratio)[0]
        selected = []
        if target_granularity == 'group':
            full_weights = self.attribute_weighting(grouped_attrs, is_group=True)
            # weights are sorted in descending order, so cut off the bottom 33% of the attributes
            if len(full_weights) > 3:
                weights = {key: val for idx, (key, val) in enumerate(full_weights.items()) if idx < len(full_weights) * 2 // 3}
            else:
                weights = full_weights
            if n is None:
                n = int(np.ceil(np.random.beta(2, 5) * len(weights) / 2))
            if weight_inverse:
                weights = {key: 1/val for key, val in weights.items()}
                weights = {key: val/sum(weights.values()) for key, val in weights.items()}
            selected_groups = np.random.choice(list(weights.keys()), n, list(weights.values()))
            # match the selected attributes with the flat_event_attrs, consider the case-insensitive matching
            for selected_group in selected_groups:
                selected_group = get_close_matches(selected_group, list(grouped_attrs.keys()), n=1, cutoff=0.8)[0]
                attrs = grouped_attrs[selected_group]
                selected_item = {'attrs': attrs, 'group_name': selected_group}
                if selected_item not in selected:
                    selected.append(selected_item)
            return {'selected': selected, 'group': True}
        elif target_granularity == 'individual':
            full_weights = self.attribute_weighting(individual_attrs, is_group=False)
            # weights are sorted in descending order, so cut off the bottom 33% of the attributes
            if len(full_weights) > 3:
                weights = {key: val for idx, (key, val) in enumerate(full_weights.items()) if idx < len(full_weights) * 2 // 3}
            else:
                weights = full_weights
            max_n_attrs = min(6, len(weights))
            if n is None:
                n = random.choice(list(range(1, max_n_attrs+1)))
            if weight_inverse:
                weights = {key: 1/val for key, val in weights.items()}
                weights = {key: val/sum(weights.values()) for key, val in weights.items()}
            try:
                selected_attr_keys = np.random.choice(list(weights.keys()), n, list(weights.values()))
            except Exception as e:
                logger.error(f"[attr-select] Cannot select attributes from weights. "
                           f"Weights: {weights}, n={n}, Error: {e}")
                raise AttributeSelectionError(f"Cannot select attributes for weights: {weights}") from e
            
            try:
                selected_attr_keys = [get_close_matches(key, list(individual_attrs.keys()), n=1, cutoff=0.2)[0] for key in selected_attr_keys]
            except (IndexError, ValueError) as e:
                logger.error(f"[attr-select] Cannot find closest match for attributes. "
                           f"Selected keys: {selected_attr_keys}, "
                           f"Available keys: {list(individual_attrs.keys())}, Error: {e}")
            #     # the attributes are sometimes a list of strings
            #     return {"error_message": "fail:sanitization - Cannot find the closest attribute for the following attributes: " + str(selected_attr_keys), "selected": None, "group": False}
            attrs = {key: individual_attrs[key] for key in selected_attr_keys}
            return {'selected': attrs, 'group': False}
        else:
            raise ValueError("The target granularity should be either 'group' or 'individual'.")
    
    def _select_and_remove_attributes(self, grouped_attrs: dict, individual_attrs: dict, method: str, 
                                    group_individual_ratio: list = None, weight_inverse: bool = False) -> dict:
        """
        Select attributes for a specific sanitization method and immediately remove them from the available pools.
        
        Args:
            grouped_attrs (dict): Dictionary of grouped attributes
            individual_attrs (dict): Dictionary of individual attributes
            method (str): Sanitization method ('abstract', 'drop', or 'keep')
            group_individual_ratio (list): Ratio for group vs individual selection
            weight_inverse (bool): Whether to use inverse weighting
            
        Returns:
            dict: Selected attributes and their group status
        """
        # Calculate number of attributes to select based on method
        if method == 'abstract':
            n = int(np.ceil(np.random.beta(2, 5) * len(grouped_attrs) / 2))
        elif method == 'drop':
            n = int(np.round(np.random.beta(2, 5) * len(grouped_attrs) / 2))
        else:  # keep
            n = int(np.floor(np.random.beta(2, 5) * len(grouped_attrs) / 2))
            n = min([n, len(grouped_attrs), len(individual_attrs)])
        
        if n <= 0:
            return {'selected': None, 'group': False}
            
        # Select attributes with appropriate ratio
        ratio = group_individual_ratio or [5, 5]  # Default to 50-50 if not specified
        selected_attrs = self.select_target_attrbutes(
            grouped_attrs, 
            individual_attrs, 
            n=n, 
            group_individual_ratio=ratio,
            weight_inverse=weight_inverse
        )

        # Immediately remove selected attributes from available pools
        # TODO: we should also remove individual attributes from the available pools
        if selected_attrs['group']:
            for group in selected_attrs['selected']:
                if group['group_name'] in grouped_attrs:
                    del grouped_attrs[group['group_name']]
                # Remove the attributes included in the group from the individual attributes pool
                for key, value in group['attrs'].items():
                    if key in individual_attrs:
                        del individual_attrs[key]
        else:
            for key, value in selected_attrs['selected'].items():
                if key in individual_attrs:
                    del individual_attrs[key]

        return selected_attrs

    def _process_individual_attributes(self, attrs_dict: dict, sanitization_method: str,
                                     selected_attrs: dict, selected_individual_attributes: dict,
                                     selected_individual_attributes_sanitization: dict) -> None:
        """
        Process individual attributes for a specific sanitization method.
        
        Args:
            attrs_dict (dict): Dictionary containing attributes for the method
            sanitization_method (str): The sanitization method to apply
            selected_attrs (dict): Selected attributes for the method
            selected_individual_attributes (dict): Dictionary to store processed attributes
            selected_individual_attributes_sanitization (dict): Dictionary to store sanitization methods
        """
        if not selected_attrs['group'] and selected_attrs['selected'] is not None:
            for key, val in selected_attrs['selected'].items():
                selected_individual_attributes[key] = val
                selected_individual_attributes_sanitization[key] = sanitization_method

    def _process_grouped_attributes(self, attrs_dict: dict, sanitization_method: str,
                                  selected_attrs: dict, selected_individual_attributes: dict,
                                  selected_individual_attributes_sanitization: dict) -> None:
        """
        Process grouped attributes for a specific sanitization method.
        
        Args:
            attrs_dict (dict): Dictionary containing attributes for the method
            sanitization_method (str): The sanitization method to apply
            selected_attrs (dict): Selected attributes for the method
            selected_individual_attributes (dict): Dictionary to store processed attributes
            selected_individual_attributes_sanitization (dict): Dictionary to store sanitization methods
        """
        if selected_attrs['group']:
            for group in selected_attrs['selected']:
                group_name = group['group_name']
                for key, val in group['attrs'].items():
                    if key in selected_individual_attributes:
                        continue
                    selected_individual_attributes[key] = val
                    selected_individual_attributes_sanitization[key] = sanitization_method
                    # Store group_name information for grouped attributes
                    if not hasattr(self, '_attribute_group_mapping'):
                        self._attribute_group_mapping = {}
                    self._attribute_group_mapping[key] = group_name

    def _get_target_lists(self, attrs_dict: dict) -> tuple:
        """
        Generate target lists for each sanitization method.
        
        Args:
            attrs_dict (dict): Dictionary containing attributes for all methods
            
        Returns:
            tuple: Lists of targets for abstract, drop, and keep methods
        """
        abstract_targets = ([val['group_name'] for val in attrs_dict['abstract']['selected']] 
                          if attrs_dict['abstract']['group'] 
                          else [f"{key.replace('_', ' ')}, {val}" for key, val in attrs_dict['abstract']['selected'].items()])
        
        drop_targets = ([val['group_name'] for val in attrs_dict['drop']['selected']] 
                       if attrs_dict['drop']['selected'] is not None and attrs_dict['drop']['group']
                       else [f"{key.replace('_', ' ')}, {val}" for key, val in attrs_dict['drop']['selected'].items()] 
                       if attrs_dict['drop']['selected'] is not None 
                       else [])
        
        keep_targets = ([val['group_name'] for val in attrs_dict['keep']['selected']] 
                       if attrs_dict['keep']['selected'] is not None and attrs_dict['keep']['group']
                       else [f"{key.replace('_', ' ')}, {val}" for key, val in attrs_dict['keep']['selected'].items()] 
                       if attrs_dict['keep']['selected'] is not None 
                       else [])
        
        return abstract_targets, drop_targets, keep_targets

    def set_sanitization_methods(self, record: dict) -> dict:
        """
        Set the sanitization methods for the attributes: drop, abstract, or keep.
        
        This method determines how each attribute should be handled during sanitization:
        - drop: completely remove the information
        - abstract: replace with generalized version
        - keep: preserve as is
        
        Args:
            record (dict): Dictionary containing grouped_attributes and individual attributes
            
        Returns:
            dict: Dictionary containing selected attributes and their sanitization methods
        """
        # Reset attribute group mapping for clean state
        self._attribute_group_mapping = {}
        
        # Create deep copies to avoid modifying original data
        grouped_attrs = deepcopy(record['grouped_attributes'])
        individual_attrs = deepcopy(record['attributes']['event'])
        if len(individual_attrs) == 0:
            # This is a data quality issue from upstream generation, not a processing error
            # Skip gracefully without alarming error messages
            return {"error_message": "skip:no_event_attrs - Record has no event attributes (data quality issue from generation)", "attrs": None, "attrs_for_abstract": None, "attrs_for_drop": None, "attrs_for_keep": None, "abstract_targets": None, "drop_targets": None, "keep_targets": None}

        # Step 1: Select attributes for abstract and drop methods
        try:
            attrs_dict = {
                'abstract': self._select_and_remove_attributes(grouped_attrs, individual_attrs, 'abstract'),
                'drop': self._select_and_remove_attributes(grouped_attrs, individual_attrs, 'drop', [7, 3])
            }
        except Exception as e:
            error_tb = traceback.format_exc()
            logger.error(f"[attr-select] Failed to select attributes for abstract/drop methods: {e}\n"
                        f"  grouped_attrs keys: {list(grouped_attrs.keys()) if grouped_attrs else 'None'}\n"
                        f"  individual_attrs keys: {list(individual_attrs.keys()) if individual_attrs else 'None'}\n"
                        f"  Traceback:\n{error_tb}")
            print(Panel(f"Error in selecting attributes for abstract and drop methods: {e}", box=box.SIMPLE_HEAD, title="Error", expand=False, style="red"))
            return {"error_message": f"fail:attr_selection - Error in selecting attributes for abstract and drop methods: {e}", "attrs": None, "attrs_for_abstract": None, "attrs_for_drop": None, "attrs_for_keep": None, "abstract_targets": None, "drop_targets": None, "keep_targets": None, "traceback": error_tb}

        # Step 2: Use LLM to find closest attributes for keep method
        # attrs_dict['keep'] = self._find_closest_attributes(
        # attrs_dict['keep'] = self._find_farthest_attributes(
        # attrs_dict['keep'] = self._find_farthest_attributes_rouge_llm(
        try:
            attrs_dict['keep'] = self._find_farthest_attributes_rouge(
                grouped_attrs, individual_attrs, 
                attrs_dict['abstract'], attrs_dict['drop']
            )
        except Exception as e:
            error_tb = traceback.format_exc()
            logger.error(f"[attr-select] Failed to find farthest attributes for keep method: {e}\n"
                        f"  abstract_attrs: {attrs_dict.get('abstract', {})}\n"
                        f"  drop_attrs: {attrs_dict.get('drop', {})}\n"
                        f"  Traceback:\n{error_tb}")
            print(Panel(f"Error in finding farthest attributes for keep method: {e}", box=box.SIMPLE_HEAD, title="Error", expand=False, style="red"))
            return {"error_message": f"fail:attr_selection - Error in finding farthest attributes for keep method: {e}", "attrs": None, "attrs_for_abstract": None, "attrs_for_drop": None, "attrs_for_keep": None, "abstract_targets": None, "drop_targets": None, "keep_targets": None, "traceback": error_tb}
        
        # Remove the selected keep attributes from available pools
        # XXX: Do we have to?
        if attrs_dict['keep']['selected'] is not None:
            if attrs_dict['keep']['group']:
                for group in attrs_dict['keep']['selected']:
                    if group['group_name'] in grouped_attrs:
                        del grouped_attrs[group['group_name']]
            else:
                for key, value in attrs_dict['keep']['selected'].items():
                    if key in individual_attrs:
                        del individual_attrs[key]

        # Step 3: Process selected attributes
        selected_individual_attributes = {}
        selected_individual_attributes_sanitization = {}
        
        # Process attributes in priority order: drop > abstract > keep
        for method in ['drop', 'abstract', 'keep']:
            self._process_individual_attributes(
                attrs_dict, method, attrs_dict[method],
                selected_individual_attributes,
                selected_individual_attributes_sanitization
            )
            self._process_grouped_attributes(
                attrs_dict, method, attrs_dict[method],
                selected_individual_attributes,
                selected_individual_attributes_sanitization
            )

        # Step 4: Create final selected attributes dictionary
        selected = {}
        for key, val in selected_individual_attributes.items():
            attr_info = {'value': val, 'sanitization': selected_individual_attributes_sanitization[key]}
            # Add group_name if this attribute came from a group
            if hasattr(self, '_attribute_group_mapping') and key in self._attribute_group_mapping:
                attr_info['group_name'] = self._attribute_group_mapping[key]
            else:
                attr_info['group_name'] = None
            selected[key] = attr_info

        # Step 5: Generate target lists
        abstract_targets, drop_targets, keep_targets = self._get_target_lists(attrs_dict)

        return {
            'attrs': selected,
            'attrs_for_abstract': attrs_dict['abstract'],
            'attrs_for_drop': attrs_dict['drop'],
            'attrs_for_keep': attrs_dict['keep'],
            'abstract_targets': abstract_targets,
            'drop_targets': drop_targets,
            'keep_targets': keep_targets
        }

    def get_sequences_by_attribute(self, extraction_keys: list, extracted_part: list, record: dict, attributes: dict) -> dict:
        """
        Extract sequences that contain each attribute and return a mapping of attribute to sequences.
        Sequences are shared objects so modifications sync across attributes.
        
        Args:
            extraction_keys (list): List of extraction keys containing sequence and attribute information
            extracted_part (list): List of extracted parts from the LLM
            record (dict): Dictionary containing text sequences to process
            attributes (dict): Dictionary of attributes to look for
            
        Returns:
            dict: Dictionary mapping each attribute to a list of shared sequence objects
        """
        # Create extracted spans by combining keys with extractions
        extracted_spans = [
            {**key, 'extraction': extraction}
            for key, extraction in zip(extraction_keys, extracted_part)
        ]

        # Initialize result dictionary which is a dictionary of attributes as keys and a list of sequence objects as values
        sequences_by_attribute = {}
        
        # Create a mapping of sequence text to sequence objects for sharing
        sequence_objects = {}
        
        # Process each extracted span, which is a list with the length of the number of attributes times the number of sequences
        for span_data in extracted_spans:
            # If there are no spans for this attribute and sequence, skip
            if span_data['extraction'].lower().strip(".") == 'none':
                continue
                
            # If there are extracted spans, get the sequence and attribute
            sequence_text = span_data['seq']
            attr = span_data['attr']
            attr_str = str(attr) # Convert attribute to string for consistent key handling
            
            # Initialize list for this attribute if not exists
            if attr_str not in sequences_by_attribute:
                sequences_by_attribute[attr_str] = [] # XXX: this is a list of sequence objects
            
            # If the sequence text is not in the sequence objects, create a new sequence object
            if sequence_text not in sequence_objects:
                # Create a new sequence object that can be modified
                sequence_obj = {
                    'text': sequence_text,
                    'terminator': span_data.get('terminator', ''),
                    'idx': span_data.get('idx', 0),
                    'spans': {}  # Dictionary to store spans for each attribute
                }
                sequence_objects[sequence_text] = sequence_obj
            else:
                # If the sequence text is in the sequence objects, use the existing sequence object
                sequence_obj = sequence_objects[sequence_text]
            
            # Compute span location and add to sequence object
            seq_start_loc = record['text'].find(sequence_text)
            if seq_start_loc != -1: # XXX: maybe the sequence text is not in the record text, because something is wrong with the segmentation of the records! if this is the case, seq_start_loc will be -1, and nothing gets added to the sequence object! Then, results in an empty spans dictionary for the attribute!
                # Handle multiple spans separated by ||
                if span_data['extraction'].lower() != 'none' or not span_data['extraction'].lower().startswith('none '):
                    spans = [span.strip() for span in span_data['extraction'].split('||')]
                    for span_text in spans:
                        # XXX: maybe the span is not getting in. 
                        span_info = self._compute_span(
                            record_text=record['text'],
                            sequence=sequence_text,
                            extraction=span_text,
                            seq_start_loc=seq_start_loc,
                            attr=attr,
                            record=record,
                            group_name=span_data.get('group_name')
                        )

                        if "error_message" in span_info:
                            # Call small model to verify if attribute is actually in the sequence
                            prompt = f"Is '{span_text}' included in the text below? Answer 'yes' or 'no' without any other text.\n\nTEXT: {sequence_text}"
                            response = self.model.interact(prompt, temperature=0.7)
                            if response.lower() == 'yes':
                                logger.warning(f"[span-extract] Span location extraction failed but attribute exists. "
                                             f"Attr: {attr}, Span: '{span_text[:50]}...', "
                                             f"Sequence preview: '{sequence_text[:100]}...'")
                                print(Panel(f"Span '{span_text}' is included in the sequence below:\n{sequence_text}\n\n*But the span location extraction failed", box=box.SIMPLE_HEAD, title="Error", expand=False, style="red"))
                                return {"error_message": f"fail:span_location - Attribute '{attr}' exists in sequence but span location extraction failed", "attr": attr, "sequence": sequence_text, "span": span_text, "extraction_keys": extraction_keys, "extracted_part": extracted_part}
                            else:
                                logger.debug(f"[span-extract] Span not found in sequence (expected). "
                                            f"Attr: {attr}, Span: '{span_text[:30]}...'")
                                continue
                        
                        if span_info['location'][0] > span_info['location'][1] or calculate_text_difference(span_text, span_info['span']) > 5:
                            logger.warning(f"[span-extract] Invalid span location or text mismatch. "
                                         f"Attr: {attr}, Location: {span_info['location']}, "
                                         f"Expected: '{span_text[:30]}...', Got: '{span_info['span'][:30]}...'")
                            print(Panel(f"Span {span_text} is weird in the sequence {sequence_text}", box=box.SIMPLE_HEAD, title="Error", expand=False, style="red"))
                            return {"error_message": f"fail:span_validation - Span location invalid or text mismatch for attr '{attr}'", "attr": attr, "sequence": sequence_text, "span": span_text, "span_info": span_info, "extraction_keys": extraction_keys, "extracted_part": extracted_part}
                        
                        if span_info and "error_message" not in span_info:
                            # Initialize spans for this attribute if not exists
                            if attr_str not in sequence_obj['spans']:
                                sequence_obj['spans'][attr_str] = []
                            
                            # Add span info to the sequence object
                            span_info_copy = copy.deepcopy(span_info)
                            span_info_copy['attr_type'] = span_data.get('attr_type')
                            span_info_copy['sanitization_option'] = span_data.get('sanitization_option')
                            sequence_obj['spans'][attr_str].append(span_info_copy)

            else:
                logger.error(f"[span-extract] Sequence not found in record text (segmentation error). "
                           f"Attr: {attr}, Sequence preview: '{sequence_text[:100]}...'")
                print(Panel(f"Sequence {sequence_text} is not in the sequence objects. Segmentation error?", box=box.SIMPLE_HEAD, title="Error", expand=False, style="red"))
                return {"error_message": f"fail:segmentation - Sequence not found in record text for attr '{attr}'", "attr": attr, "sequence": sequence_text, "extraction_keys": extraction_keys, "extracted_part": extracted_part}

            # Add shared sequence object if there is span info for this attribute
            if sequence_obj not in sequences_by_attribute[attr_str] and len(sequence_obj['spans'].get(attr_str, [])) > 0:
                sequences_by_attribute[attr_str].append(sequence_obj)
        
        # Merge overlapping spans for each attribute in each sequence
        for sequence_obj in sequence_objects.values():
            for attr_str, spans in sequence_obj['spans'].items():
                if len(spans) > 1:
                    # Sort spans by location for consistent merging
                    spans.sort(key=lambda x: x['location'][0])
                    
                    # Merge overlapping spans
                    i = 0
                    while i < len(spans):
                        j = i + 1
                        while j < len(spans):
                            if self._spans_overlap(spans[i]['location'], spans[j]['location']):
                                merged_span = self._merge_spans(spans[i], spans[j])
                                if merged_span is not None:
                                    spans[i] = merged_span
                                    spans.pop(j)
                                else:
                                    j += 1
                            else:
                                j += 1
                        i += 1

            # Iterate through the sequences_by_attribute and check if the sequence has no spans for the attributes
        for attr, sequences in sequences_by_attribute.items():
            for sequence in sequences:
                if len(sequence['spans'].get(attr, [])) == 0:
                    logger.warning(f"[span-extract] Sequence has no spans for attribute. "
                                 f"Attr: {attr}, Sequence preview: '{sequence['text'][:100]}...'")
                    print(Panel(f"Sequence {sequence['text']} has no spans for attribute {attr}", box=box.SIMPLE_HEAD, title="Error", expand=False, style="red"))
                    return {"error_message": f"fail:span_extraction - Sequence has no spans for attribute '{attr}'", "attr": attr, "sequence": sequence['text']}

        # Remove sequences that have no spans for the attributes
        targets_to_remove = []
        for attr, sequences in sequences_by_attribute.items():
            if len(sequences) == 0:
                targets_to_remove.append(attr)
        if len(targets_to_remove) > 0:
            logger.warning(f"[span-extract] Removing {len(targets_to_remove)} attributes with no spans: {targets_to_remove}")
            print(Panel(f"Removing {len(targets_to_remove)} attributes from sequences_by_attribute because they have no spans", box=box.SIMPLE_HEAD, title="Error", expand=False, style="red"))
            return {"error_message": f"fail:span_extraction - No spans found for {len(targets_to_remove)} attributes: {targets_to_remove}", "attrs_without_spans": targets_to_remove, "sequences": sequences}
        for attr in targets_to_remove:
            del sequences_by_attribute[attr]
        
        return sequences_by_attribute

    def _build_abstraction_strategy_prompt(self, attr_type: str, attr_value: str, sequences: list, group_name: str = None) -> str:
        """
        Get the abstraction strategy for the attribute
        """
        sequences_str = "\n\n".join(sequences)
        
        # Add group information to the prompt if available
        if group_name:
            group_info = f"{group_name.replace('_', ' ')}, "
        else:
            group_info = ""

        attr_type_str = attr_type.replace("_", " ")
        attr_value_str = attr_value.replace("_", " ")

        prompt = f"""You are an **Attribute Generalizer**.

Based on the following given sequences, specifically describe how we should abstract '{attr_value_str}' ({attr_type_str}) in an instruction without any additional comments.

RULES:
1. Use specific, actionable verbs (e.g., "abstract", "redact", "mask", "generalize", "replace with", "convert to") that abstract specific identifying details but preserve the meaning
2. For IDs, sex, gender, race, codes, names, emails, or URLs, instead of abstracting, mask or redact the value.
3. If the value is a specific time range (e.g., 11:20-13:00), abstract the time range (e.g., from before lunch to after lunch). If the time range is day-specific, abstract the day. If the time range is month-specific, abstract the month. If the time range is year-specific, abstract the year.
4. If the value is a simple category, drop or redact the category. For example, high, medium, low, or yes, no, should be redacted instead of abstracted.
5. If the value is a location/affiliation, make sure you hide other information that can be used to identify the location/affiliation, such as the city, state, email address, or organization name.
6. Ensure the instruction can be applied consistently across different contexts in the following sequences.
7. Consider the group context when abstracting - if this attribute belongs to a specific group, ensure the abstraction is consistent with other attributes in the same group.
8. Make sure the instruction is self-contained by including the attribute type and value in the instruction.

SEQUENCES:
{sequences_str}

Instruction:
"""

        return prompt

    def _build_abstraction_prompt(self, attr_type: str, attr_value: str, sequence: str, strategy: str, group_name: str = None, spans_str: str = None) -> str:
        """
        Build a prompt for sanitizing the sequence using the given strategy for the attribute
        """
        # Add group information to the prompt if available
        group_info = ""
        if group_name:
            group_info = f"{group_name.replace('_', ' ')}, "

        attr_type_str = attr_type.replace("_", " ")
        prompt = f"""You are a **Text Sequence Sanitizer**.
        
Abstract the '{group_info}{attr_type_str}, {attr_value}' in the sequence using the following strategy. I will provide the spans of the attribute in the sequence. Make **no other changes** to the sequence. Return only the rewritten sequence—no tags, no commentary.
{strategy}

Spans that need to be dropped from the sequence (if any):
{spans_str}

TARGET SEQUENCE TO SANITIZE:
{sequence}"""

        return prompt

    def _build_drop_prompt(self, attr_type: str, attr_value: str, sequence: str, group_name: str = None, spans_str: str = None) -> str:
        """
        Drop the attribute value from the sequence
        """
        # Add group information to the prompt if available
        group_info = ""
        if group_name:
            group_info = f"{group_name.replace('_', ' ')}, "
        
        attr_type_str = attr_type.replace("_", " ")
        prompt = f"""You are a **Minimal Remover**.

I will provide the spans of the attribute in a sequence. Drop the '{group_info}{attr_type_str}, {attr_value}' from the sequence below without any additional comments.

RULES:
1. Drop the attribute value '{attr_value}' from the sequence, try to minimize the changes to the sequence.
2. Smooth the surrounding sentence so it reads naturally. For example, if the attribute is a value in key-value pairs, drop the value and the key too.
3. Return only the rewritten sequence. Do not remove other unrelated parts of the sequence.

Spans that need to be dropped from the sequence (if any):
{spans_str}

TARGET SEQUENCE:
{sequence}"""

        return prompt

    def sanitize_attributes(self, strategies_by_attribute: dict) -> dict:
        """
        Sanitize the attributes
        """
        def build_sanitization_prompt(attr: str, strategy: str) -> str:
            """
            Build a sanitization prompt for the attribute
            """
            prompt = f"""You are a **Text Sequence Sanitizer**.

Abstract '{attr}' using the following strategy. Return only the rewritten text—no tags, no commentary.
{strategy}"""
            return prompt

        sanitization_prompts = []
        for attr, strategy in strategies_by_attribute.items():
            sanitization_prompt = build_sanitization_prompt(attr, strategy)
            sanitization_prompts.append(sanitization_prompt)
        sanitized_attributes = self.model.batch_interact(sanitization_prompts, temperature=0.7)

        sanitized_attributes_mapping = {}
        for attr, sanitized_attr in zip(strategies_by_attribute.keys(), sanitized_attributes):
            sanitized_attributes_mapping[attr] = sanitized_attr

        return sanitized_attributes_mapping

    def sanitize_sequences_by_attribute(self, sequences_by_attribute: dict, record: dict) -> dict:
        """
        Sanitize the sequences by attribute:
        - replace the spans with the sanitized spans by rewritting the sequence text
        - sanitized sequences are shared objects so modifications sync across attributes
        """
        strategy_prompts = []
        abstract_attrs = []
        drop_attrs = []
        
        # Separate attributes by sanitization method
        for attr, sequences in sequences_by_attribute.items():
            if len(sequences) == 0:
                logger.warning(f"[sanitize-seq] No sequences found for attribute '{attr}'. Skipping.")
                print(Panel(f"No sequences found for attribute {attr}", box=box.SIMPLE_HEAD, title="Error", expand=False, style="red"))
                continue

            sequence_texts = [sequence['text'] for sequence in sequences]
            try:
                attr_type = sequences[0]['spans'][attr][0]['attr_type']
                sanitization_option = sequences[0]['spans'][attr][0]['sanitization_option']
            except (KeyError, IndexError, TypeError) as e:
                logger.warning(f"[sanitize-seq] Failed to get attr_type/sanitization_option for attr '{attr}'. "
                             f"Spans keys: {list(sequences[0].get('spans', {}).keys())}, Error: {e}")
                print(Panel(f"Error in getting attr_type or sanitization_option for attribute {attr}", box=box.SIMPLE_HEAD, title="Error", expand=False, style="red"))
                continue

            try:
                if isinstance(attr_type, list):
                    attr_type_str = ", ".join(attr_type).replace("_", " ")
                else:
                    attr_type_str = attr_type.replace("_", " ")
            except (TypeError, AttributeError) as e:
                logger.warning(f"[sanitize-seq] Failed to convert attr_type to string for attr '{attr}'. "
                             f"attr_type={attr_type}, Error: {e}")
                print(Panel(f"Error in getting attr_type_str for attribute {attr}", box=box.SIMPLE_HEAD, title="Error", expand=False, style="red"))
            if 'abstract' in sanitization_option:
                # Get group_name from the span information
                group_name = sequences[0]['spans'][attr][0].get('group_name')
                strategy_prompt = self._build_abstraction_strategy_prompt(attr_type_str, attr, sequence_texts, group_name)
                strategy_prompts.append(strategy_prompt)
                abstract_attrs.append((attr_type_str, attr))
            elif 'drop' in sanitization_option:
                drop_attrs.append((attr_type_str, attr))

        # batch interact with the prompts to get strategies for abstract attributes only
        strategies = self.model.batch_interact(strategy_prompts, temperature=1) if strategy_prompts else []

        # Create a mapping of attributes to their strategies (None for drop attributes)
        strategies_by_attribute = {}
        for attr_type_str, attr in abstract_attrs:
            if "Instruction:" in strategies[abstract_attrs.index((attr_type_str, attr))]:
                abstract_strategy = strategies[abstract_attrs.index((attr_type_str, attr))].split("Instruction:")[1].strip("*").strip()
            else:
                abstract_strategy = strategies[abstract_attrs.index((attr_type_str, attr))]
            strategies_by_attribute[f"{attr_type_str}, {attr}"] = abstract_strategy
        for attr_type_str, attr in drop_attrs:
            # strategies_by_attribute[f"{attr_type_str}, {attr}"] = f"Drop '{attr}' ({attr_type_str}) from the text."
            strategies_by_attribute[f"{attr_type_str}, {attr}"] = f"Drop the {attr_type_str} information '{attr}' from the text."

        # Process all attributes (both abstract and drop)
        for attr, sequences in sequences_by_attribute.items():
            sanitization_prompts = []
            sanitization_option = sequences[0]['spans'][attr][0]['sanitization_option']
            attr_type = sequences[0]['spans'][attr][0]['attr_type']
            if isinstance(attr_type, list):
                attr_type_str = ", ".join(attr_type).replace("_", " ")
            else:
                attr_type_str = attr_type.replace("_", " ")
            
            if 'abstract' in sanitization_option:
                try:
                    strategy = strategies_by_attribute[f"{attr_type_str}, {attr}"]
                except KeyError as e:
                    logger.warning(f"[sanitize-seq] Strategy not found for attribute '{attr_type_str}: {attr}'. "
                                 f"Available strategies: {list(strategies_by_attribute.keys())}")
                    print(Panel(f"Error in getting strategy for attribute {attr_type_str}: {attr}", box=box.SIMPLE_HEAD, title="Error", expand=False, style="red"))
                    continue
                for sequence in sequences:
                    # Get group_name from the span information
                    group_name = sequence['spans'][attr][0].get('group_name')
                    spans = [span_obj['span'] for span_obj in sequence['spans'][attr]]
                    spans_str = " || ".join(spans)
                    sanitization_prompt = self._build_abstraction_prompt(attr_type_str, attr, sequence['text'], strategy, group_name, spans_str)
                    sanitization_prompts.append(sanitization_prompt)
            elif 'drop' in sanitization_option:
                for sequence in sequences:
                    spans = [span_obj['span'] for span_obj in sequence['spans'][attr]]
                    spans_str = " || ".join(spans)
                    sanitization_prompt = self._build_drop_prompt(attr_type_str, attr, sequence['text'], sequence['spans'][attr][0].get('group_name'), spans_str)
                    sanitization_prompts.append(sanitization_prompt)
            else:
                raise ValueError(f"Invalid sanitization option: {sanitization_option}")

            try:
                sanitized_sequences = self.model.batch_interact(sanitization_prompts, temperature=0.7)
            except Exception as e:
                logger.warning(f"[sanitize-seq] First batch_interact attempt failed for attr '{attr}': {e}. Retrying...")
                time.sleep(1)
                try:
                    sanitized_sequences = self.model.batch_interact(sanitization_prompts, temperature=0.7)
                except Exception as retry_e:
                    logger.error(f"[sanitize-seq] Retry failed for attr '{attr}': {retry_e}")
                    raise SequenceSanitizationError(f"Failed to sanitize sequences for attr '{attr}': {retry_e}") from retry_e

            # TODO: add a check to see if the sanitized sequences are valid. For example, if the sanitized sequence includes unwanted texts, such as additional comments or denial of the sanitization, the sanitization is invalid.

            for sequence, sanitized_sequence in zip(sequences, sanitized_sequences):
                if "Sanitized Text:" in sanitized_sequence:
                    sanitized_sequence = sanitized_sequence.split("Sanitized Text:")[1].strip()
                if "Sanitized text:" in sanitized_sequence:
                    sanitized_sequence = sanitized_sequence.split("Sanitized text:")[1].strip()
                if "sanitized text:" in sanitized_sequence:
                    sanitized_sequence = sanitized_sequence.split("sanitized text:")[1].strip()
                sequence['text'] = sanitized_sequence

        if self.args.print:
            print(Panel("Sequence Sanitization Done!", box=box.SIMPLE_HEAD, expand=False, style="grey53"))

        return {'sequences_by_attribute': sequences_by_attribute, 'strategies_by_attribute': strategies_by_attribute}

    def rebuild_record_w_sanitized_seqs(self, segmented_sequences: list, sanitized_sequences_by_attribute: dict) -> dict:
        """
        Rebuild the record from the sanitized sequences
        - fix the punctuation and leading whitespaces of the sanitized sequences to match the original sequences
        - replace the original sequences with the sanitized sequences
        - rebuild the record from the sanitized sequences
        """

        def fix_punctuation(orig: str, text: str) -> str:
            """Ensure the sanitized text ends with similar punctuation as the original."""
            if text.strip() == "":
                return text

            if orig.endswith(".") and not text.endswith("."):
                text += "."
            if orig.endswith("?") and not text.endswith("?"):
                text += "?"
            if orig.endswith("!") and not text.endswith("!"):
                text += "!"
            # Remove trailing period if original didn't end with one.
            if not orig.endswith(".") and text.endswith("."):
                text = text.rstrip(".")
            return text

        def preserve_leading_whitespace(orig: str, text: str) -> str:
            """Preserve the leading whitespace from the original sequence."""
            if text.strip() == "":
                return text

            if orig.startswith(" "):
                whitespace_count = len(orig) - len(orig.lstrip())
                text = " " * whitespace_count + text.strip()
            return text

        # get all the unique sequences from the sanitized sequences by attribute
        sanitized_sequences = []
        for attr, sequences in sanitized_sequences_by_attribute.items():
            for sequence in sequences:
                sanitized_sequences.append((sequence['idx'], sequence['text']))
        sanitized_sequences = list(set(sanitized_sequences))
        sanitized_sequences.sort(key=lambda x: x[0])

        # Create detailed mapping dictionary
        sequence_sanitization_mapping = {}
        
        # replace the original sequences with the sanitized sequences
        for idx, sanitized_seq in sanitized_sequences:
            original_seq = segmented_sequences[idx]['seq']
            sanitized_seq = fix_punctuation(original_seq, sanitized_seq)
            sanitized_seq = preserve_leading_whitespace(original_seq, sanitized_seq)
            segmented_sequences[idx]['seq'] = sanitized_seq
            
            # Build detailed mapping for this sequence
            sequence_sanitization_mapping[idx] = {
                'original_sequence': original_seq,
                'sanitized_sequence': sanitized_seq,
                'target_attributes': [],
                'strategies_used': [],
                'spans': []
            }
            
            # Collect all attributes, strategies, and spans for this sequence
            for attr, sequences in sanitized_sequences_by_attribute.items():
                for sequence in sequences:
                    if sequence['idx'] == idx:
                        # Add target attribute
                        if attr not in sequence_sanitization_mapping[idx]['target_attributes']:
                            sequence_sanitization_mapping[idx]['target_attributes'].append(attr)
                        
                        # Add strategy
                        if attr in sequence['spans']:
                            for span in sequence['spans'][attr]:
                                strategy = span.get('sanitization_option', 'unknown')
                                if strategy not in sequence_sanitization_mapping[idx]['strategies_used']:
                                    sequence_sanitization_mapping[idx]['strategies_used'].append(strategy)

                        # Add spans
                        if attr in sequence['spans']:
                            for span in sequence['spans'][attr]:
                                span_info = {
                                    'attribute': attr,
                                    'span_text': span.get('span', ''),
                                    'location': span.get('location', (0, 0)),
                                    'confidence': span.get('confidence', 0.0),
                                    'attr_type': span.get('attr_type', ''),
                                    'sanitization_option': span.get('sanitization_option', ''),
                                    'group_name': span.get('group_name', '')
                                }
                                sequence_sanitization_mapping[idx]['spans'].append(span_info)

        sanitized_record = self.rebuild_record(segmented_sequences)
        return {
            'text': sanitized_record, 
            'sanitized_sequences_by_attribute': sanitized_sequences_by_attribute,
            'sequence_sanitization_mapping': sequence_sanitization_mapping
        }

    def sanitize_privasis_data(self, datapoint, idx) -> list:
        """
        Sanitization phase:
        1. segment the record into sequences: segment_record()
        2. select the target attributes for sanitization and set the sanitization methods (drop, abstract, keep) for each of the target attributes: set_sanitization_methods()
        3. extract the spans where the target attributes are included: extract_spans()
        4. sanitize the sequences by dropping the attributes and abstracting the sequences with the abstracted attributes: sanitize_sequences_by_attribute()
        5. rebuild the record from the sanitized sequences: rebuild_record_w_sanitized_seqs()
        """
        record = datapoint['record']

        # Preprocess the records and select the target attributes
        segmented_record = self.decompose_record(record)
        record['sequences'] = copy.deepcopy(segmented_record['sequences'])
        record['text'] = segmented_record['text']

        attrs_and_sanitization = self.set_sanitization_methods(record)
        if "error_message" in attrs_and_sanitization:
            # Ensure error_message is first key for readability in error logs
            error_msg = attrs_and_sanitization.pop('error_message')
            return {"error_message": error_msg, **attrs_and_sanitization, 'folder_num': idx, **datapoint}
        target_attributes = {key: val for key, val in attrs_and_sanitization['attrs'].items() if val['sanitization'] in ['drop', 'abstract']}

        # Extract the spans where the attributes are included and identify the included attributes
        sequences_by_attribute = self.extract_spans(record, target_attributes)
        if "error_message" in sequences_by_attribute:
            # Ensure error_message is first key for readability in error logs
            error_msg = sequences_by_attribute.pop('error_message')
            return {"error_message": error_msg, **sequences_by_attribute, 'attributes': target_attributes, 'folder_num': idx, **datapoint}

        if sequences_by_attribute == {}:
            logger.warning(f"[span-extract] No spans extracted for record idx={idx}. "
                         f"Target attrs: {list(target_attributes.keys())}")
            print(cf.yellow("No spans are extracted"))
            return {"error_message": "fail:span_extraction - No valid spans extracted", **sequences_by_attribute, 'attributes': target_attributes, 'folder_num': idx, **datapoint}

        try:
            sanitized_seqs = self.sanitize_sequences_by_attribute(copy.deepcopy(sequences_by_attribute), record)
            # add keep attributes to the strategies_by_attribute
            if attrs_and_sanitization['attrs_for_keep']['selected']:
                keep_targets = attrs_and_sanitization['attrs_for_keep']['selected']
                if isinstance(keep_targets, list):
                    for _keep_target in keep_targets:
                        for attr_type, attr in _keep_target['attrs'].items():
                            attr_type_str = attr_type.replace("_", " ")
                            sanitized_seqs['strategies_by_attribute'][f"{attr_type_str}, {attr}"] = f"Keep the {attr_type_str} information '{attr}' in the text."
                else:
                    for attr_type, attr in keep_targets.items():
                        attr_type_str = attr_type.replace("_", " ")
                        sanitized_seqs['strategies_by_attribute'][f"{attr_type_str}, {attr}"] = f"Keep the {attr_type_str} information '{attr}' in the text."
        except Exception as e:
            error_tb = traceback.format_exc()
            logger.error(f"[sanitize] Sanitization failed for record idx={idx}: {e}\n"
                        f"  Target attrs: {list(target_attributes.keys())}\n"
                        f"  Record preview: '{record['text'][:200]}...'\n"
                        f"  Traceback:\n{error_tb}")
            print(Panel(f"Sanitization Failed: {e}", box=box.SIMPLE_HEAD, title="Error", expand=False, style="red"))
            return {"error_message": f"fail:sanitization - {type(e).__name__}: {e}", "traceback": error_tb, **sequences_by_attribute, 'attributes': target_attributes, 'folder_num': idx, **datapoint}
        sanitized_sequences_by_attribute = sanitized_seqs['sequences_by_attribute']
        strategies_by_attribute = sanitized_seqs['strategies_by_attribute']

        sanitized_record = self.rebuild_record_w_sanitized_seqs(segmented_record['sequences'].copy(), sanitized_sequences_by_attribute)

        sanitized_record['sanitization_info'] = attrs_and_sanitization
        sanitized_record['sanitization_info']['attrs_to_keep'] =  {key: val for key, val in attrs_and_sanitization['attrs'].items() if val['sanitization'] == "keep"}
        sanitized_record['sanitization_info']['individual_target_attributes'] = target_attributes
        sanitized_record['sanitization_info']['strategies'] = strategies_by_attribute
        del sanitized_record['sanitization_info']['attrs']

        sanitized_result = {'sanitized_record': sanitized_record, 'folder_num': idx, **datapoint}

        if self.args.print:
            print(Panel(record['text'], box=box.SIMPLE_HEAD, title="Original Record", expand=False, style="grey53"))
            print(Panel(json.dumps(target_attributes, indent=4), box=box.SIMPLE_HEAD, title="Target Attributes: ", expand=False, style="grey53"))
            print(Panel(json.dumps(strategies_by_attribute, indent=4), box=box.SIMPLE_HEAD, title="Sanitization Strategies by Attribute: ", expand=False, style="green"))
            print(Panel(sanitized_record['text'], box=box.SIMPLE_HEAD, title="Sanitized Record", expand=False, style="blue"))

        return sanitized_result

    def extract_spans(self, record: dict, attributes: dict) -> dict:
        """
        Extract spans of text containing sensitive information from a record.
        
        Args:
            record (dict): Dictionary containing text sequences to process
            attributes (dict): Dictionary of sensitive information to look for
            
        Returns:
            dict: Dictionary containing extracted spans and their metadata
            
        Raises:
            ValueError: If record or attributes are invalid
        """
        if not record or not attributes:
            raise ValueError("Record and attributes must not be empty")
            
        extraction_prompts = []
        extraction_keys = []
        
        # Pre-calculate max length for efficiency
        max_length = max([len(seq['seq'].split()) for seq in record['sequences']]) + 200
        
        for sequence_obj in record['sequences']:
            sequence = sequence_obj['seq']
            terminator = sequence_obj['terminator']
            
            # Skip invalid sequences
            if sequence in ["", " ", "\n", "---"]:
                continue
                
            for attr_type, _attr in attributes.items():
                try:
                    attr = _attr['value']
                    sanitization_option = _attr['sanitization']
                    group_name = _attr.get('group_name')  # Get group_name from attribute info
                    
                    # Handle different attribute types
                    if isinstance(attr, list):
                        if attr and isinstance(attr[0], dict):
                            # Handle list of dictionaries
                            for a in attr:
                                for k, v in a.items():
                                    self._add_extraction_prompt(
                                        sequence, terminator, v, k,
                                        sequence_obj, attr_type, sanitization_option, group_name,
                                        extraction_prompts, extraction_keys
                                    )
                        else:
                            # Handle simple list
                            for a in attr:
                                self._add_extraction_prompt(
                                    sequence, terminator, a, attr_type,
                                    sequence_obj, None, sanitization_option, group_name,
                                    extraction_prompts, extraction_keys
                                )
                    else:
                        # Handle single value
                        self._add_extraction_prompt(
                            sequence, terminator, attr, attr_type,
                            sequence_obj, None, sanitization_option, group_name,
                            extraction_prompts, extraction_keys
                        )
                except Exception as e:
                    logger.warning(f"[span-extract] Error processing attribute '{attr_type}': {e}. "
                                 f"Attr value preview: {str(_attr)[:50]}...")
                    continue
                    
        # Process all prompts
        if extraction_prompts:
            try:
                extracted_part = self.model.batch_interact(extraction_prompts, temperature=0.7, max_tokens=max_length)
            except Exception as e:
                logger.warning(f"[span-extract] First batch_interact attempt failed: {e}. Retrying...")
                try:
                    extracted_part = self.model.batch_interact(extraction_prompts, temperature=0.7, max_tokens=max_length)
                except Exception as retry_e:
                    logger.error(f"[span-extract] Retry failed: {retry_e}")
                    return {'error_message': f"fail:span_extraction - Model interaction failed after retry: {retry_e}"}
            
            sequences_by_attribute = self.get_sequences_by_attribute(extraction_keys, extracted_part, record, attributes)
            if "error_message" in sequences_by_attribute:
                logger.warning(f"[span-extract] Extraction failed: {sequences_by_attribute['error_message']}")
                print(Panel(f"Span Extraction Failed: {sequences_by_attribute['error_message']}", box=box.SIMPLE_HEAD, title="Error", expand=False, style="red"))
            if self.args.print:
                print(Panel("Span Extraction Done!", box=box.SIMPLE_HEAD, expand=False, style="grey53"))
            return sequences_by_attribute
        
        logger.warning(f"[span-extract] No valid prompts generated. Attributes: {list(attributes.keys())}")
        return {'error_message': f"fail:span_extraction - No valid prompts generated for {len(attributes)} attributes"}
        
    def _add_extraction_prompt(self, sequence: str, terminator: str, attr_value: any, attr_type: str,
                             sequence_obj: dict, upper_attr_type: str, sanitization_option: str, group_name: str,
                             extraction_prompts: list, extraction_keys: list) -> None:
        """
        Helper method to add extraction prompts and keys.
        
        Args:
            sequence (str): The text sequence to process
            terminator (str): The sequence terminator
            attr_value (any): The attribute value to look for
            attr_type (str): The type of attribute
            sequence_obj (dict): The sequence object containing metadata
            upper_attr_type (str): The parent attribute type if applicable
            sanitization_option (str): The sanitization method to apply
            group_name (str): The group name of the attribute
            extraction_prompts (list): List to store extraction prompts
            extraction_keys (list): List to store extraction keys
        """
        extraction_prompt = self.build_extraction_prompt(sequence, terminator, attr_value, attr_type)
        extraction_prompts.append(extraction_prompt)
        extraction_keys.append({
            **sequence_obj,
            'upper_attr_type': upper_attr_type,
            'attr_type': attr_type,
            'attr': str(attr_value),
            'sanitization_option': sanitization_option,
            'group_name': group_name
        })

    def build_extraction_prompt(self, sequence: str, terminator: str, attr_value: str, attr_type: str) -> str:
        """
        Build a prompt for extraction based on the target text, attribute value, and attribute type.
        
        Args:
            sequence (str): The text sequence to analyze
            terminator (str): The sequence terminator
            attr_value (str): The attribute value to look for
            attr_type (str): The type of attribute
            
        Returns:
            str: A well-structured prompt for the LLM to extract sensitive information
        """
        target_text = sequence + terminator
        
        # Add specific instructions based on attribute type
        type_specific_instructions = {
            'name': "Look for full names, partial names, titles, user handles, pronouns that could identify the person.",
            'email': "Look for email addresses, including partial matches or references to email domains.",
            'phone': "Look for phone numbers, including formatted and unformatted versions.",
            'address': "Look for street addresses, cities, states, or postal codes.",
            'date': "Look for dates in various formats, including relative dates and age references.",
            'id': "Look for identification numbers, account numbers, or reference codes.",
            'url': "Look for website URLs, including partial matches or domain references.",
            'location': "Look for specific locations, landmarks, or geographical references.",
            'organization': "Look for company names, department names, or organizational references.",
            'medical': "Look for medical conditions, treatments, or health-related information.",
            'financial': "Look for financial information, amounts, or transaction details."
        }
        
        # Get specific instructions for the attribute type
        specific_instruction = type_specific_instructions.get(attr_type.lower(), 
            "Look for any information that could reveal or infer the target value.")
        
        prompt = (
            f"Task: Identify sensitive information in the target text.\n\n"
            f"Target Information:\n"
            f"- Type: {attr_type}\n"
            f"- Value: {attr_value}\n\n"
            f"Instructions:\n"
            f"1. {specific_instruction}\n"
            f"2. Consider both direct mentions and indirect references (e.g., age can be inferred from date of birth, location can be inferred from address, etc.).\n"
            f"3. Look for variations in how the information might be expressed.\n"
            f"4. Consider the context - only extract if the information is actually being revealed.\n"
            f"5. Extract ALL spans that contain the sensitive information, not just the most specific one.\n\n"
            f"6. Do not output overlapping spans.\n\n"
            f"7. If the information is not in the target text, output ONLY the word 'None'.\n"
            f"Output Format:\n"
            f"- If you find the information, output each span on a new line, separated by '||'.\n"
            f"- Each span should be the exact text that contains the sensitive information.\n"
            f"- If you don't find the information, output ONLY the word 'None'.\n"
            f"- Do not include any explanations or additional text.\n\n"
            f"Example output for multiple spans:\n"
            f"John Smith||Mr. Smith||he\n\n"
            f"Target text to analyze:\n{target_text}"
        ).strip()
        
        return prompt

    def _spans_overlap(self, span1: tuple, span2: tuple) -> bool:
        """
        Check if two spans overlap at all.
        
        Args:
            span1 (tuple): (start, end) location of first span
            span2 (tuple): (start, end) location of second span
            
        Returns:
            bool: True if spans overlap at all, False otherwise
        """
        return not (span1[1] <= span2[0] or span2[1] <= span1[0])

    def _merge_spans(self, span1: dict, span2: dict) -> dict:
        """
        Merge two overlapping spans into a single span, but only if they share the same sanitization option.
        
        Args:
            span1 (dict): First span with location and metadata
            span2 (dict): Second span with location and metadata
            
        Returns:
            dict: Merged span with combined location and metadata, or None if spans shouldn't be merged
        """
        # Check if spans share the same sanitization option
        span1_options = set(span1['sanitization_option'] if isinstance(span1['sanitization_option'], list) else [span1['sanitization_option']])
        span2_options = set(span2['sanitization_option'] if isinstance(span2['sanitization_option'], list) else [span2['sanitization_option']])
        
        if not span1_options.intersection(span2_options):
            return None  # Don't merge if sanitization options don't overlap
            
        # Calculate merged location
        start = min(span1['location'][0], span2['location'][0])
        end = max(span1['location'][1], span2['location'][1])
        
        # Get the full text of the merged span by combining both spans
        merged_text = ""
        if span1['location'][0] <= span2['location'][0]:
            # span1 starts first
            merged_text = span1['span']
            if span1['location'][1] < span2['location'][1]:
                # Add the non-overlapping part from span2
                overlap_start = span1['location'][1] - span2['location'][0]
                if overlap_start < 0:
                    # Add gap between spans
                    gap = " " * abs(overlap_start)
                    merged_text += gap
                    overlap_start = 0
                merged_text += span2['span'][overlap_start:]
        else:
            # span2 starts first
            merged_text = span2['span']
            if span2['location'][1] < span1['location'][1]:
                # Add the non-overlapping part from span1
                overlap_start = span2['location'][1] - span1['location'][0]
                if overlap_start < 0:
                    # Add gap between spans
                    gap = " " * abs(overlap_start)
                    merged_text += gap
                    overlap_start = 0
                merged_text += span1['span'][overlap_start:]
        
        # Calculate merged confidence (weighted average by span length)
        len1 = span1['location'][1] - span1['location'][0]
        len2 = span2['location'][1] - span2['location'][0]
        total_len = len1 + len2
        merged_confidence = (span1['confidence'] * len1 + span2['confidence'] * len2) / total_len
        
        # Combine attributes and their metadata
        attributes = []
        if isinstance(span1['attr'], list):
            attributes.extend(span1['attr'])
        else:
            attributes.append(span1['attr'])
        if isinstance(span2['attr'], list):
            attributes.extend(span2['attr'])
        else:
            attributes.append(span2['attr'])
            
        # Combine attribute types
        attr_types = []
        if isinstance(span1['attr_type'], list):
            attr_types.extend(span1['attr_type'])
        else:
            attr_types.append(span1['attr_type'])
        if isinstance(span2['attr_type'], list):
            attr_types.extend(span2['attr_type'])
        else:
            attr_types.append(span2['attr_type'])
            
        # Deduplicate attributes and their types while maintaining the order
        seen_attrs = set()
        deduped_attrs = []
        deduped_types = []
        
        for attr, attr_type in zip(attributes, attr_types):
            # Convert attribute to string for comparison if it's not already
            attr_str = str(attr).lower()
            if attr_str not in seen_attrs:
                seen_attrs.add(attr_str)
                deduped_attrs.append(attr)
                deduped_types.append(attr_type)
            
        # Use the common sanitization options
        sanitization_options = span1_options.pop()
            
        return {
            'attr': deduped_attrs,
            'span': merged_text,
            'location': (start, end),
            'confidence': merged_confidence,
            'upper_attr_type': span1.get('upper_attr_type'),
            'attr_type': deduped_types,
            'sanitization_option': sanitization_options,
            'merged_from': [span1, span2],  # Track which spans were merged
            'group_name': span1.get('group_name')  # Add group_name to merged span
        }

    def _compute_span(self, record_text: str, sequence: str, extraction: str, seq_start_loc: int, attr: any, record: dict = None, group_name: str = None) -> dict:
        """
        Compute the location of the span within the record text based on the extraction result.
        
        Args:
            record_text (str): The full text of the record
            sequence (str): The sequence containing the span
            extraction (str): The extracted text from the LLM
            seq_start_loc (int): The starting location of the sequence in the record
            attr (any): The attribute value being searched for
            record (dict): Optional record metadata
            group_name (str): The group name of the attribute
            
        Returns:
            dict: Span information including location, confidence score, and metadata
        """
        seq_lower = sequence.lower()

        # If the model returns something other than "none", try to locate it
        if not extraction.lower().startswith("none"):
            # Calculate confidence score based on match quality
            confidence_score = 0.0
            span_info = None
            
            # Try original extraction as is first (highest confidence)
            index = seq_lower.find(extraction.lower())
            if index != -1:
                confidence_score = 1.0
                span_info = {
                    'attr': attr,
                    'cleaned_span': extraction,
                    'location': (seq_start_loc + index, seq_start_loc + index + len(extraction)),
                    'span': record_text[seq_start_loc + index:seq_start_loc + index + len(extraction)],
                    'confidence': confidence_score,
                    'group_name': group_name
                }
            else:
                # Try clean_extraction (medium confidence)
                clean_extraction = extraction.strip().strip("'").strip('"').lower().strip()
                index = seq_lower.find(clean_extraction)
                if index != -1:
                    confidence_score = 0.95
                    span_info = {
                        'attr': attr,
                        'cleaned_span': clean_extraction,
                        'location': (seq_start_loc + index, seq_start_loc + index + len(clean_extraction)),
                        'span': record_text[seq_start_loc + index:seq_start_loc + index + len(clean_extraction)],
                        'confidence': confidence_score,
                        'group_name': group_name
                    }
                else:
                    # Try alternative cleaning (lowest confidence)
                    clean_extraction_alt = clean_extraction.replace("*", "").replace('"', "").strip(".").strip()
                    index = seq_lower.find(clean_extraction_alt)
                    if index != -1:
                        confidence_score = 0.9
                        span_info = {
                            'attr': attr,
                            'cleaned_span': clean_extraction_alt,
                            'location': (seq_start_loc + index, seq_start_loc + index + len(clean_extraction_alt)),
                            'span': record_text[seq_start_loc + index:seq_start_loc + index + len(clean_extraction_alt)],
                            'confidence': confidence_score,
                            'group_name': group_name
                        }
            
            if span_info:
                return span_info
            else:
                logger.debug(f"[span-extract] Extracted span not found in sequence. "
                           f"Attr: {attr}, Extraction: '{extraction[:50]}...', "
                           f"Sequence preview: '{sequence[:80]}...', "
                           f"Record type: {record.get('type', '') if record else 'N/A'}")
                print(Panel(
                    f"Sequence: {seq_lower}\nSpan: {clean_extraction}\nRecord type: {record.get('type', '') if record else ''}",
                    title="Error: The extracted span is not found in the sequence.",
                    box=box.SIMPLE_HEAD,
                    expand=False,
                    style="red"
                ))
                return {'error_message': f"fail:span_location - Extracted span not found in sequence for attr '{attr}'", 'sequence': sequence, 'extraction': extraction, 'attr': attr}
        else:
            # When extraction is "None", attempt to locate the attribute value directly
            if isinstance(attr, list):
                try:
                    attr_str = " ".join(attr).lower()
                except Exception:
                    attr_str = " ".join([str(item) for item in attr]).lower()
            else:
                attr_str = str(attr).lower()

            index = seq_lower.find(attr_str)
            if index != -1:
                return {
                    'attr': attr,
                    'span': record_text[seq_start_loc + index: seq_start_loc + index + len(attr_str)],
                    'location': (seq_start_loc + index, seq_start_loc + index + len(attr_str)),
                    'confidence': 0.3,  # Lower confidence for direct attribute matching
                    'group_name': group_name
                }
            logger.debug(f"[span-extract] Attr value not found in sequence (extraction was 'None'). "
                        f"Attr: '{attr_str[:30]}...', Sequence preview: '{sequence[:80]}...'")
            return {'error_message': f"fail:span_extraction - Attr value '{attr}' not found in sequence", 'sequence': sequence, 'extraction': extraction, 'attr': attr}

    def _process_target_attributes_for_instruction(self, targets, sanitization_strategies, grouped_attributes, action_type):
        """
        Helper function to process target attributes and build sanitization strategies string.
        
        Args:
            targets: List of target attributes
            sanitization_strategies: Dictionary of sanitization strategies
            grouped_attributes: Dictionary of grouped attributes
            action_type: Type of action ('abstract', 'drop', or 'keep')
            
        Returns:
            str: Built sanitization strategies string for the given targets
        """
        strategies_str = ""
        
        for target in targets:
            if target in sanitization_strategies:
                strategies_str += f"- {sanitization_strategies[target]}\n"
            else:  # which means the attribute is grouped
                if random.random() < 0.5:
                    if action_type == 'abstract':
                        strategies_str += f"- Generalize the information about {target} in the text.\n"
                    elif action_type == 'drop':
                        strategies_str += f"- Drop the information about {target} from the text.\n"
                    elif action_type == 'keep':
                        strategies_str += f"- Keep the information about {target} in the text.\n"
                else:  # which means we add the example attributes to the instruction
                    try:
                        group_attrs = [att.replace("_", " ") for att in grouped_attributes[target]]
                    except (KeyError, TypeError) as e:
                        logger.warning(f"[instruction-gen] Failed to get group attributes for target '{target}': {e}. "
                                      f"Available groups: {list(grouped_attributes.keys())}")
                        print(Panel(f"Error in getting group attributes for attribute {target}", box=box.SIMPLE_HEAD, title="Error", expand=False, style="red"))
                        return {'error_message': f"fail:instruction_generation - Error getting group attributes for '{target}': {e}", 'attribute': target}
                    group_attrs_str = ", ".join(group_attrs[:-1]) + ", and " + group_attrs[-1]
                    # add the example attributes to the instruction
                    if action_type == 'abstract':
                        strategies_str += f"- Generalize the information about {target} (e.g., {group_attrs_str}) in the text.\n"
                    elif action_type == 'drop':
                        strategies_str += f"- Drop the information about {target} (e.g., {group_attrs_str}) from the text.\n"
                    elif action_type == 'keep':
                        strategies_str += f"- Keep the information about {target} (e.g., {group_attrs_str}) in the text.\n"
        
        return strategies_str

    def build_instructions(self, sanitized_result):
        """
        Set the base instruction for the sanitization process.
        """
        sanitization_info = sanitized_result['sanitized_record']['sanitization_info']
        sanitization_strategies = sanitization_info['strategies']
        abstraction_target = sanitization_info['abstract_targets']
        drop_target = sanitization_info['drop_targets']
        keep_target = sanitization_info['keep_targets']
        original_record = sanitized_result['record']['text']
        sanitized_record = sanitized_result['sanitized_record']['text']
        grouped_attributes = sanitized_result['record']['grouped_attributes']

        if random.random() < 0.5:
            sanitization_strategies_str = ""
            
            # Process abstract targets
            abstract_str = self._process_target_attributes_for_instruction(abstraction_target, sanitization_strategies, grouped_attributes, 'abstract')
            if isinstance(abstract_str, dict) and 'error_message' in abstract_str:
                return abstract_str
            sanitization_strategies_str += abstract_str
            
            # Process drop targets
            drop_str = self._process_target_attributes_for_instruction(drop_target, sanitization_strategies, grouped_attributes, 'drop')
            if isinstance(drop_str, dict) and 'error_message' in drop_str:
                return drop_str
            sanitization_strategies_str += drop_str
            
            # Process keep targets
            if random.random() < 0.25:  # we only occasionally add the keep targets to the instruction
                keep_str = self._process_target_attributes_for_instruction(keep_target, sanitization_strategies, grouped_attributes, 'keep')
                if isinstance(keep_str, dict) and 'error_message' in keep_str:
                    return keep_str
                sanitization_strategies_str += keep_str
        else:
            sanitization_strategies_str = "\n".join([f"- {v}" for k, v in sanitization_strategies.items()])

        sanitization_strategies_str = sanitization_strategies_str.strip()
        
        if not sanitization_strategies_str:
            logger.warning(f"[instruction-gen] Empty sanitization strategies string. "
                         f"Abstract targets: {abstraction_target}, Drop targets: {drop_target}")
            return {'error_message': 'fail:instruction_gen - Empty sanitization strategies'}
        
        prompt = f"""Use the following sanitization strategies to build a very specific instruction for the sanitization process. Do not add any additional comments.\n\nSanitization strategies:\n{sanitization_strategies_str}"""
        
        try:
            smoothed_instruction = self.model.interact(prompt, temperature=1.2, max_tokens=5000)
        except Exception as e:
            error_tb = traceback.format_exc()
            logger.error(f"[instruction-gen] Model interaction failed: {e}\n{error_tb}")
            return {'error_message': f'fail:instruction_gen - Model interaction failed: {e}', 'traceback': error_tb}

        if self.args.print:
            print(Panel(sanitization_strategies_str, box=box.SIMPLE_HEAD, title="Sanitization Strategies Concatenated", expand=False, style="green"))
            print(Panel(smoothed_instruction, box=box.SIMPLE_HEAD, title="Smoothed Instruction", expand=False, style="green"))

        return {'base_instruction': sanitization_strategies_str, 'smoothed_instruction': smoothed_instruction}

    def get_under_sanitized_attrs(self, sanitized_result):
        n_guesses = 3
        sanitized_record = sanitized_result['sanitized_record']

        # generate questions for the validation
        question_header = "Given the following text, please answer the question below."
        sanitized_text = sanitized_record['text']
        prompts = []
        attrs = []
        for attr, info in sanitized_record['selected_attributes'].items():
            attr_str = attr.replace("_", " ")
            if info['sanitization'] == 'drop':
                # generate a question for the attribute
                # question = f"Question: Generate 2 possible specific {attr_str} that can be inferred from the text. Answer in an ordered list. Make it specific and do not add extra comments."
                question = f"Question: What is the specific {attr_str} that can be inferred from the text? Make a guess even if it's not included in the given text. Make it specific and do not add extra comments.\nAnswer:"
            elif info['sanitization'] == 'abstract':
                # question = f"Question: Generate 2 possible specific {attr_str} that can be inferred from the text using '{info['abstracted_value']}'. Answer in an ordered list. Make it specific and do not add extra comments."
                question = f"Question: What is the specific {attr_str} that can be inferred from the text? Make a guess. Hint: '{info['abstracted_value']}'. Make the answer specific and concise, and do not add extra comments.\nAnswer:"

            if info['sanitization'] in ['drop', 'abstract']:
                prompt = f"{question_header}\n\nText:\n{sanitized_text}\n\n{question}"
                for _ in range(n_guesses):
                    attrs.append({attr: info})
                    prompts.append(prompt)
            # print(Panel(question, box=box.SIMPLE_HEAD, title="Validation Question", expand=False, style="purple"))
        _results = self.model.batch_interact(prompts, temperature=1.0)
        # results = [capture_and_parse_ordered_list(result) for result in _results]
        results = [result.lower() for result in _results]

        # check the validation results
        under_sanitized_attrs = []
        for attr_info, guesses in zip(attrs, results):
            original_attr = list(attr_info.values())[0]['value']
            if type(original_attr) is str:
                if original_attr.lower() in guesses:
                    under_sanitized_attrs.append(attr_info)
            elif type(original_attr) is list:
                for idx, a in enumerate(original_attr):
                    _attr_info = deepcopy(attr_info)
                    key = list(attr_info.keys())[0]
                    _attr_info[key]['value'] = a
                    _attr_info[key]['abstracted_value'] = attr_info[key]['abstracted_value'][idx]
                    if a.lower() in guesses:
                        under_sanitized_attrs.append({key + f"_{idx + 1}": _attr_info[key]})
        
        return under_sanitized_attrs

    def get_model_inference_from_original_record(self, target_attributes: dict, text: str) -> str:
        question_header = "Given the following text, please answer the question below."
        prompts = []
        attrs = []
        
        for attr, info in target_attributes.items():
            attr_str = attr.replace("_", " ")
            
            if info['sanitization'] == 'drop':
                question = f"Question: What is the specific '{attr_str}' information that can be inferred from the text? Make a guess even if it's not included in the given text. Make it specific and do not add extra comments.\nAnswer:"
            elif info['sanitization'] == 'abstract':
                question = f"Question: What is the specific '{attr_str}' information that can be inferred from the text? Make a guess. Make the answer specific and concise, and do not add extra comments.\nAnswer:"
            elif info['sanitization'] == 'keep':
                question = f"Question: What is the specific '{attr_str}' information that can be inferred from the text? Make the answer specific and concise, and do not add extra comments.\nAnswer:"
            else:
                continue  # Skip attributes that weren't sanitized
                
            if info['sanitization'] in ['drop', 'abstract', 'keep']:
                prompt = f"{question_header}\n\nText:\n{text}\n\n{question}"
                attrs.append({attr: info})
                prompts.append(prompt)
        
        # Get LLM responses
        results = self.model.batch_interact(prompts, temperature=1.0)
        # results = [result for result in results]
        return results

    def save(self, output_dir: str, result: dict, visual=False):
        """Save result to file with thread-safe locking."""
        with self._file_lock:
            with open(output_dir, "a") as fp:
                if visual:
                    fp.write(json.dumps(result, indent=4) + "\n")
                else:
                    fp.write(json.dumps(result) + "\n")

    def get_last_save_idx(self) -> int:
        logs = read_logs(self.output_file)
        error_logs = read_logs(self.error_file)
        if logs is not None:
            last_save_idx = logs['folder_num'].iloc[-1]
        else:
            last_save_idx = -1
        if error_logs is not None:
            last_error_idx = error_logs['folder_num'].iloc[-1]
            if last_error_idx > last_save_idx:
                last_save_idx = last_error_idx
        return last_save_idx

    def _process_datapoint(self, idx: int, privasis_datapoint) -> Tuple[int, dict, bool]:
        """Process a single datapoint and return the result.
        
        This method is designed to be called from multiple threads.
        
        Args:
            idx: Index of the datapoint
            privasis_datapoint: Data to sanitize
            
        Returns:
            Tuple of (index, result_dict, is_error)
        """
        sanitization_result = self.sanitize_privasis_data(privasis_datapoint, idx)
        if 'error_message' in sanitization_result:
            return idx, sanitization_result, True
        
        instructions = self.build_instructions(sanitization_result)
        if 'error_message' in instructions:
            return idx, instructions, True
        
        sanitization_result['base_instruction'] = instructions['base_instruction']
        sanitization_result['smoothed_instruction'] = instructions['smoothed_instruction']
        
        # Process target attributes
        target_attrs = sanitization_result['sanitized_record']['sanitization_info']['individual_target_attributes']
        inference_from_original_record = self.get_model_inference_from_original_record(target_attrs, sanitization_result['record']['text'])
        for i, (attr_key, attr_info) in enumerate(sanitization_result['sanitized_record']['sanitization_info']['individual_target_attributes'].items()):
            if i < len(inference_from_original_record):
                attr_info['inference_from_original_record'] = inference_from_original_record[i]
            else:
                attr_info['inference_from_original_record'] = None

        attrs_to_keep = sanitization_result['sanitized_record']['sanitization_info']['attrs_to_keep']
        inference_from_original_record = self.get_model_inference_from_original_record(attrs_to_keep, sanitization_result['record']['text'])
        for i, (attr_key, attr_info) in enumerate(attrs_to_keep.items()):
            if i < len(inference_from_original_record):
                attr_info['inference_from_original_record'] = inference_from_original_record[i]
            else:
                attr_info['inference_from_original_record'] = None
        sanitization_result['sanitized_record']['sanitization_info']['attrs_to_keep'] = attrs_to_keep

        return idx, sanitization_result, False

    def _worker_process_chunk(self, worker_id: int, chunk: list) -> dict:
        """Process a chunk of datapoints assigned to a specific worker.
        
        Args:
            worker_id: ID of the worker (0-indexed)
            chunk: List of (idx, row) tuples to process
            
        Returns:
            Dict with counts of processed, errors, and skipped items
        """
        processed = 0
        errors = 0
        skipped = 0
        
        for idx, row in tqdm(chunk, desc=f"Worker {worker_id}", position=worker_id, leave=True):
            try:
                idx, result, is_error = self._process_datapoint(idx, row)
                if is_error:
                    error_msg = result.get('error_message', 'Unknown error')
                    # Differentiate between skip (data quality) and fail (processing error)
                    if error_msg.startswith('skip:'):
                        # Data quality issue - just log quietly and save to errors file
                        logger.debug(f"[worker-{worker_id}] Skipped idx={idx}: {error_msg}")
                        tqdm.write(f"  Worker {worker_id} skipped index {idx}: {error_msg.split(' - ', 1)[-1]}")
                        skipped += 1
                    else:
                        # Actual processing error - log with more detail
                        logger.warning(f"[worker-{worker_id}] Error at idx={idx}: {error_msg}")
                        tqdm.write(f"  Worker {worker_id} error at index {idx}: {error_msg}")
                        errors += 1
                    self.save(self.error_file, result)
                else:
                    self.save(self.output_file, result)
                    processed += 1
            except Exception as e:
                error_tb = traceback.format_exc()
                logger.error(f"[worker-{worker_id}] Unhandled exception at idx={idx}: {type(e).__name__}: {e}\n{error_tb}")
                tqdm.write(f"Worker {worker_id} exception at index {idx}: {e}")
                self.save(self.error_file, {
                    "error": f"[thread-exception] {type(e).__name__}: {e}",
                    "traceback": error_tb,
                    "idx": idx,
                    "worker_id": worker_id
                })
                errors += 1
        
        logger.info(f"[worker-{worker_id}] Completed. Processed={processed}, Errors={errors}, Skipped={skipped}")
        return {"worker_id": worker_id, "processed": processed, "errors": errors, "skipped": skipped}

    def run(self):
        """
        Run sanitization pipeline
        """
        self.privasis = load_privasis(self.args.privasis_data_id)
        last_save_idx = self.get_last_save_idx()
        
        # Filter datapoints that haven't been processed yet
        datapoints_to_process = [(idx, row) for idx, row in self.privasis.iterrows() if idx > last_save_idx]
        
        if not datapoints_to_process:
            print("All datapoints already processed.")
            return
        
        # Use multithreading if num_workers > 1 and using vLLM server
        if self.num_workers > 1 and self.vllm_server_url:
            print(f"Running with {self.num_workers} parallel workers (vLLM server: {self.vllm_server_url})")
            print(f"Total datapoints to process: {len(datapoints_to_process)}")
            
            # Split datapoints into chunks for each worker
            total_items = len(datapoints_to_process)
            chunk_size = (total_items + self.num_workers - 1) // self.num_workers  # Ceiling division
            chunks = []
            for i in range(self.num_workers):
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, total_items)
                if start_idx < total_items:
                    chunks.append(datapoints_to_process[start_idx:end_idx])
            
            print(f"Split into {len(chunks)} chunks: {[len(c) for c in chunks]}")
            print()
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit one task per worker, each processing its own chunk
                futures = {
                    executor.submit(self._worker_process_chunk, worker_id, chunk): worker_id 
                    for worker_id, chunk in enumerate(chunks)
                }
                
                # Collect results from all workers
                results = []
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        worker_id = futures[future]
                        error_tb = traceback.format_exc()
                        logger.error(f"[run] Worker {worker_id} failed with exception: {type(e).__name__}: {e}\n{error_tb}")
                        print(f"Worker {worker_id} failed with exception: {e}")
                        results.append({"worker_id": worker_id, "processed": 0, "errors": 1, "skipped": 0})
            
            # Print summary
            print()
            total_processed = sum(r["processed"] for r in results)
            total_errors = sum(r["errors"] for r in results)
            total_skipped = sum(r["skipped"] for r in results)
            logger.info(f"[run] Completed. Total: Processed={total_processed}, Errors={total_errors}, Skipped={total_skipped}")
            print(f"Summary: Processed={total_processed}, Errors={total_errors}, Skipped={total_skipped}")
        else:
            # Single-threaded execution (original behavior)
            if self.num_workers > 1 and not self.vllm_server_url:
                print("Warning: --num-workers > 1 requires --vllm-server-url. Running single-threaded.")
            
            for idx, privasis_datapoint in tqdm(datapoints_to_process, total=len(datapoints_to_process), desc="Sanitizing records"):
                sanitization_result = self.sanitize_privasis_data(privasis_datapoint, idx)
                if 'error_message' in sanitization_result:
                    error_msg = sanitization_result['error_message']
                    # Differentiate between skip (data quality) and fail (processing error)
                    if error_msg.startswith('skip:'):
                        # Data quality issue - just log quietly
                        tqdm.write(f"  Skipped index {idx}: {error_msg.split(' - ', 1)[-1]}")
                    else:
                        # Actual processing error - show panel
                        print(Panel("Error: The sanitization pipeline is not successful for folder number " + str(idx) + "\n" + error_msg, box=box.SIMPLE_HEAD, expand=False, style="purple"))
                        print()
                    self.save(self.error_file, sanitization_result)
                else:
                    instructions = self.build_instructions(sanitization_result)
                    if 'error_message' in instructions:
                        print(Panel("Error: The instruction generation pipeline is not successful for folder number " + str(idx) + "\n" + instructions['error_message'], box=box.SIMPLE_HEAD, expand=False, style="purple"))
                        print()
                        self.save(self.error_file, instructions)
                        continue
                    sanitization_result['base_instruction'] = instructions['base_instruction']
                    sanitization_result['smoothed_instruction'] = instructions['smoothed_instruction']
                    
                    # XXX:
                    target_attrs = sanitization_result['sanitized_record']['sanitization_info']['individual_target_attributes']
                    inference_from_original_record = self.get_model_inference_from_original_record(target_attrs, sanitization_result['record']['text'])
                    # Add the inference results to each individual target attribute
                    for i, (attr_key, attr_info) in enumerate(sanitization_result['sanitized_record']['sanitization_info']['individual_target_attributes'].items()):
                        if i < len(inference_from_original_record):
                            attr_info['inference_from_original_record'] = inference_from_original_record[i]
                        else:
                            attr_info['inference_from_original_record'] = None

                    attrs_to_keep = sanitization_result['sanitized_record']['sanitization_info']['attrs_to_keep']
                    inference_from_original_record = self.get_model_inference_from_original_record(attrs_to_keep, sanitization_result['record']['text'])
                    for i, (attr_key, attr_info) in enumerate(attrs_to_keep.items()):
                        if i < len(inference_from_original_record):
                            attr_info['inference_from_original_record'] = inference_from_original_record[i]
                        else:
                            attr_info['inference_from_original_record'] = None
                    sanitization_result['sanitized_record']['sanitization_info']['attrs_to_keep'] = attrs_to_keep

                    self.save(self.output_file, sanitization_result)

    def save_clean_csv(self):
        df = pd.read_json(self.output_file, lines=True)
        clean = []
        for idx, row in df.iterrows():
            strategies = row['sanitized_record']['sanitization_info']['strategies']
            del row['sanitized_record']['sanitization_info']['strategies']
            sanitized_sequences = copy.deepcopy(row['sanitized_record']['sequence_sanitization_mapping'])
            for key, value in sanitized_sequences.items():
                del sanitized_sequences[key]['spans']
            clean.append(
                {
                    'grouped_attributes': json.dumps(row['record']['grouped_attributes'], indent=4),
                    'target_attributes': json.dumps(row['sanitized_record']['sanitization_info'], indent=4),
                    'situation': row['situation']['text'],
                    'sanitized_sequences': json.dumps(sanitized_sequences, indent=4),
                    'record_type': row['record']['type'],
                    'record': row['record']['text'],
                    'strategies': json.dumps(strategies, indent=4),
                    'sanitized_record': row['sanitized_record']['text'],
                    'base_instruction': row['base_instruction'],
                    'smoothed_instruction': row['smoothed_instruction'],
                }
            )
        clean_df = pd.DataFrame(clean)
        clean_df.to_csv(self.output_file.removesuffix(".jsonl") + ".csv", index=False)

    def _find_farthest_attributes_rouge(self, grouped_attrs: dict, individual_attrs: dict,
                                  abstract_attrs: dict, drop_attrs: dict, 
                                  model_name: str = "gpt-4.1-mini") -> dict:
        """
        Use token overlap to find the farthest attributes to the selected abstract and drop attributes.
        
        Args:
            grouped_attrs (dict): Available grouped attributes
            individual_attrs (dict): Available individual attributes
            abstract_attrs (dict): Selected attributes for abstraction
            drop_attrs (dict): Selected attributes for dropping
            model_name (str): LLM model to use (not used in this implementation)
            
        Returns:
            dict: Selected keep attributes that are farthest from abstract/drop attributes
        """
        
        # Step 1: Concatenate all values from abstract_attrs and drop_attrs
        selected_values = []
        
        # Extract values from abstract_attrs
        if abstract_attrs['selected'] is not None:
            if abstract_attrs['group']:
                for group in abstract_attrs['selected']:
                    for key, val in group['attrs'].items():
                        if isinstance(val, list):
                            selected_values.extend([str(v) for v in val])
                        else:
                            selected_values.append(str(val))
            else:
                for key, val in abstract_attrs['selected'].items():
                    if isinstance(val, list):
                        selected_values.extend([str(v) for v in val])
                    else:
                        selected_values.append(str(val))
        
        # Extract values from drop_attrs
        if drop_attrs['selected'] is not None:
            if drop_attrs['group']:
                for group in drop_attrs['selected']:
                    for key, val in group['attrs'].items():
                        if isinstance(val, list):
                            selected_values.extend([str(v) for v in val])
                        else:
                            selected_values.append(str(val))
            else:
                for key, val in drop_attrs['selected'].items():
                    if isinstance(val, list):
                        selected_values.extend([str(v) for v in val])
                    else:
                        selected_values.append(str(val))
        
        # Concatenate all selected values
        selected_text = " ".join(selected_values)
        
        # Step 2: Prepare available attributes for scoring
        available_attrs = {}
        
        # Add individual attributes
        for key, val in individual_attrs.items():
            if isinstance(val, list):
                available_attrs[f"<individual>_{key}"] = " ".join([str(v) for v in val])
            else:
                available_attrs[f"<individual>_{key}"] = str(val)
        
        # Add grouped attributes
        for group_name, group_attrs in grouped_attrs.items():
            group_values = []
            for key, val in group_attrs.items():
                if isinstance(val, list):
                    group_values.extend([str(v) for v in val])
                else:
                    group_values.append(str(val))
            available_attrs[f"<group>_{group_name}"] = " ".join(group_values)
        
        # Step 3: Compute ROUGE scores for each available attribute
        rouge = Rouge()
        attr_scores = {}
        
        for attr_name, attr_text in available_attrs.items():
            try:
                scores = rouge.get_scores(selected_text, attr_text)
                
                # Use only ROUGE-1 recall score
                avg_score = scores[0]['rouge-1']['r']
                
            except Exception as e:
                avg_score = 100

            attr_scores[attr_name] = {
                'score': avg_score,
                'text': attr_text,
                'type': 'individual' if attr_name.startswith('<individual>_') else 'group',
                'original_key': attr_name.replace('<individual>_', '').replace('<group>_', '')
            }
        
        # Step 4: Sort by score (ascending) and select bottom 3-4 values
        sorted_attrs = sorted(attr_scores.items(), key=lambda x: x[1]['score'])
        
        # Select only attributes with ROUGE score of zero (no overlap with selected attributes)
        selected_farthest = [(attr_name, attr_info) for attr_name, attr_info in sorted_attrs if attr_info['score'] == 0.0]

        # only select 1 or 2 attributes
        if len(selected_farthest) >= 2:
            num_to_select = random.choice([1, 2])
            selected_farthest = random.sample(selected_farthest, num_to_select)

        # If no attributes have zero score, fall back to selecting the bottom 1-2 attributes
        if not selected_farthest:
            # num_to_select = min(2, len(sorted_attrs))
            # selected_farthest = sorted_attrs[:num_to_select]
            return {'selected': None, 'group': False}
        
        # Step 5: Check for overlap with abstract_attrs and drop_attrs
        final_selected = []
        selected_keys = set()
        
        # Extract keys from abstract and drop attributes for overlap checking
        abstract_keys = set()
        drop_keys = set()
        
        if abstract_attrs['selected'] is not None:
            if abstract_attrs['group']:
                for group in abstract_attrs['selected']:
                    abstract_keys.update(group['attrs'].keys())
            else:
                abstract_keys.update(abstract_attrs['selected'].keys())
        
        if drop_attrs['selected'] is not None:
            if drop_attrs['group']:
                for group in drop_attrs['selected']:
                    drop_keys.update(group['attrs'].keys())
            else:
                drop_keys.update(drop_attrs['selected'].keys())
        
        # Filter out overlapping attributes
        for attr_name, attr_info in selected_farthest:
            original_key = attr_info['original_key']
            if original_key not in abstract_keys and original_key not in drop_keys:
                final_selected.append((attr_name, attr_info))
                selected_keys.add(original_key)
        
        # Step 6: Create the result structure
        if not final_selected:
            return {'selected': None, 'group': False}
        
        # Determine if we should return grouped or individual attributes
        group_count = sum(1 for _, info in final_selected if info['type'] == 'group')
        
        result = {
            'group': True if random.random() < 0.5 and group_count > 0 else False,
            'selected': None
        }
        
        if result['group']:
            # Return grouped attributes
            selected_groups = []
            for attr_name, attr_info in final_selected:
                if attr_info['type'] == 'group':
                    group_name = attr_info['original_key']
                    selected_groups.append({
                        'attrs': grouped_attrs[group_name],
                        'group_name': group_name
                    })
            result['selected'] = selected_groups
        else:
            # Return individual attributes
            selected_individual = {}
            for attr_name, attr_info in final_selected:
                if attr_info['type'] == 'individual':
                    key = attr_info['original_key']
                    try:
                        selected_individual[key] = individual_attrs[key]
                    except:
                        # from IPython import embed; embed(colors='neutral')  # XXX DEBUG - disabled for batch runs
                        pass
            result['selected'] = selected_individual
            result['group'] = False
        
        return result

    def _find_farthest_attributes_llm(self, grouped_attrs: dict, individual_attrs: dict, 
                                abstract_attrs: dict, drop_attrs: dict) -> dict:
        """
        Use LLM to find the farthest attributes to the selected abstract and drop attributes.
        
        Args:
            grouped_attrs (dict): Available grouped attributes
            individual_attrs (dict): Available individual attributes
            abstract_attrs (dict): Selected attributes for abstraction
            drop_attrs (dict): Selected attributes for dropping
            
        Returns:
            dict: Selected keep attributes that are farthest to abstract/drop attributes
        """
        # Prepare the selected attributes for the prompt
        abstract_targets = []
        drop_targets = []
        
        # Extract abstract targets
        if abstract_attrs['selected'] is not None:
            if abstract_attrs['group']:
                for group in abstract_attrs['selected']:
                    abstract_targets.append(f"- {group['group_name']}")
                    # Append individual attributes within the group
                    for key, val in group['attrs'].items():
                        abstract_targets.append(f"  - {key}: {val}")
            else:
                for key, val in abstract_attrs['selected'].items():
                    abstract_targets.append(f"- {key}: {val}")
        
        # Extract drop targets
        if drop_attrs['selected'] is not None:
            if drop_attrs['group']:
                for group in drop_attrs['selected']:
                    drop_targets.append(f"- {group['group_name']}")
                    # Append individual attributes within the group
                    for key, val in group['attrs'].items():
                        drop_targets.append(f"  - {key}: {val}")
            else:
                for key, val in drop_attrs['selected'].items():
                    drop_targets.append(f"- {key}: {val}")
        
        # Prepare available attributes for selection
        available_grouped = list(grouped_attrs.keys())
        available_individual = []
        for key, val in individual_attrs.items():
            available_individual.append(f"- {key}: {val}")
        
        # Build the prompt
        prompt = f"""You are an **Attribute Irrelevance Analyzer**.

I have selected some attributes for sanitization (abstraction and dropping). I need you to find the most irrelevant attributes from the available pool that should be kept (not sanitized). 
The attributes that are farthest from the selected abstract/drop attributes should be selected.

SELECTED ATTRIBUTES FOR ABSTRACTION:
{chr(10).join([f"{target}" for target in abstract_targets]) if abstract_targets else "None"}

SELECTED ATTRIBUTES FOR DROPPING:
{chr(10).join([f"{target}" for target in drop_targets]) if drop_targets else "None"}

AVAILABLE INDIVIDUAL ATTRIBUTES:
{chr(10).join(available_individual)}

TASK: Select the attributes that are most irrelevant in nature, sensitivity, or context to the selected abstract/drop attributes. These will be kept as-is.

RULES:
1. Choose attributes that have the most irrelevant privacy sensitivity levels
2. Consider the most irrelevant semantic similarity
3. Consider the most irrelevant contextual similarity
4. Do not select attributes that include or involve any of the abstract/drop attributes
5. Do not select attributes that has word overlap with the abstract/drop attributes
6. Choose 1-2 attributes total (if possible)

Output format: Return only the selected attribute names, one per line, without any additional text or formatting.

Selected attributes to keep:"""

        # Get LLM response
        response = self.model.interact(prompt, temperature=0.3)
        
        # Parse the response to extract selected attributes
        selected_keep_attrs = []
        for line in response.strip().split('\n'):
            attr_name = line.strip().strip('- ').strip()
            if attr_name:
                selected_keep_attrs.append(attr_name)
        
        # Match selected attributes with available attributes
        matched_grouped = []
        matched_individual = {}
        
        for selected_attr in selected_keep_attrs:
            # Try to match with grouped attributes first
            matched = False
            for group_name in available_grouped:
                if selected_attr.lower() in group_name.lower() or group_name.lower() in selected_attr.lower():
                    matched_grouped.append(group_name)
                    matched = True
                    break
            
            # If not matched with grouped, try individual attributes
            if not matched:
                for ind_attr, ind_val in individual_attrs.items():
                    if selected_attr.lower() in ind_attr.lower() or ind_attr.lower() in selected_attr.lower():
                        matched_individual[ind_attr] = ind_val
                        matched = True
                        break
        
        # Create the result structure
        result = {
            'group': len(matched_grouped) > len(matched_individual),
            'selected': None
        }
        
        if result['group'] and matched_grouped:
            # Return grouped attributes
            selected_groups = []
            for group_name in matched_grouped:
                selected_groups.append({
                    'attrs': grouped_attrs[group_name],
                    'group_name': group_name
                })
            result['selected'] = selected_groups
        elif matched_individual:
            # Return individual attributes
            result['selected'] = matched_individual
            result['group'] = False

        return result

    def _find_farthest_attributes_rouge_llm(self, grouped_attrs: dict, individual_attrs: dict, 
                                            abstract_attrs: dict, drop_attrs: dict) -> dict:
        """
        Use LLM to find the farthest attributes to the selected abstract and drop attributes among the available attributes that has ROUGE score of zero.
        
        Args:
            grouped_attrs (dict): Available grouped attributes
            individual_attrs (dict): Available individual attributes
            abstract_attrs (dict): Selected attributes for abstraction
            drop_attrs (dict): Selected attributes for dropping
            
        Returns:
            dict: Selected keep attributes that are farthest from abstract/drop attributes
        """
        # Step 1: Concatenate all values from abstract_attrs and drop_attrs
        selected_values = []
        
        # Extract values from abstract_attrs
        if abstract_attrs['selected'] is not None:
            if abstract_attrs['group']:
                for group in abstract_attrs['selected']:
                    for key, val in group['attrs'].items():
                        if isinstance(val, list):
                            selected_values.extend([str(v) for v in val])
                        else:
                            selected_values.append(str(val))
            else:
                for key, val in abstract_attrs['selected'].items():
                    if isinstance(val, list):
                        selected_values.extend([str(v) for v in val])
                    else:
                        selected_values.append(str(val))
        
        # Extract values from drop_attrs
        if drop_attrs['selected'] is not None:
            if drop_attrs['group']:
                for group in drop_attrs['selected']:
                    for key, val in group['attrs'].items():
                        if isinstance(val, list):
                            selected_values.extend([str(v) for v in val])
                        else:
                            selected_values.append(str(val))
            else:
                for key, val in drop_attrs['selected'].items():
                    if isinstance(val, list):
                        selected_values.extend([str(v) for v in val])
                    else:
                        selected_values.append(str(val))
        
        # Concatenate all selected values
        selected_text = " ".join(selected_values)
        
        # Step 2: Prepare available attributes for scoring
        available_attrs = {}
        
        # Add individual attributes
        for key, val in individual_attrs.items():
            if isinstance(val, list):
                available_attrs[f"<individual>_{key}"] = " ".join([str(v) for v in val])
            else:
                available_attrs[f"<individual>_{key}"] = str(val)
        
        # Add grouped attributes
        for group_name, group_attrs in grouped_attrs.items():
            group_values = []
            for key, val in group_attrs.items():
                if isinstance(val, list):
                    group_values.extend([str(v) for v in val])
                else:
                    group_values.append(str(val))
            available_attrs[f"<group>_{group_name}"] = " ".join(group_values)
        
        # Step 3: Compute ROUGE scores and filter attributes with zero score
        rouge = Rouge()
        zero_score_attrs = {}
        
        for attr_name, attr_text in available_attrs.items():
            try:
                scores = rouge.get_scores(selected_text, attr_text)
                
                # Use only ROUGE-1 recall score
                avg_score = scores[0]['rouge-1']['r']
                
                # Only keep attributes with zero ROUGE score
                if avg_score == 0.0:
                    zero_score_attrs[attr_name] = {
                        'score': avg_score,
                        'text': attr_text,
                        'type': 'individual' if attr_name.startswith('<individual>_') else 'group',
                        'original_key': attr_name.replace('<individual>_', '').replace('<group>_', '')
                    }
            except Exception as e:
                logger.debug(f"[attr-select] Error computing ROUGE score for '{attr_name}': {e}")
                continue
        
        # If no attributes have zero score, return None
        if not zero_score_attrs:
            return {'selected': None, 'group': False}
        
        # Step 4: Use LLM to find the farthest attributes among zero-score attributes
        # Prepare the selected attributes for the prompt
        abstract_targets = []
        drop_targets = []
        
        # Extract abstract targets
        if abstract_attrs['selected'] is not None:
            if abstract_attrs['group']:
                for group in abstract_attrs['selected']:
                    abstract_targets.append(f"- {group['group_name']}")
                    # Append individual attributes within the group
                    for key, val in group['attrs'].items():
                        abstract_targets.append(f"  - {key}: {val}")
            else:
                for key, val in abstract_attrs['selected'].items():
                    abstract_targets.append(f"- {key}: {val}")
        
        # Extract drop targets
        if drop_attrs['selected'] is not None:
            if drop_attrs['group']:
                for group in drop_attrs['selected']:
                    drop_targets.append(f"- {group['group_name']}")
                    # Append individual attributes within the group
                    for key, val in group['attrs'].items():
                        drop_targets.append(f"  - {key}: {val}")
            else:
                for key, val in drop_attrs['selected'].items():
                    drop_targets.append(f"- {key}: {val}")
        
        # Prepare available zero-score attributes for selection
        available_zero_score = []
        for attr_name, attr_info in zero_score_attrs.items():
            original_key = attr_info['original_key']
            if attr_info['type'] == 'individual':
                available_zero_score.append(f"- {original_key}: {attr_info['text']}")
            else:
                available_zero_score.append(f"- {original_key}: {attr_info['text']}")
        
        # Build the prompt
        prompt = f"""You are an **Attribute Irrelevance Analyzer**.

I have selected some attributes for sanitization (abstraction and dropping). Among the available attributes that have NO text overlap with the selected attributes (ROUGE score = 0), I need you to find the most irrelevant attributes.

SELECTED ATTRIBUTES FOR ABSTRACTION:
{chr(10).join([f"{target}" for target in abstract_targets]) if abstract_targets else "None"}

SELECTED ATTRIBUTES FOR DROPPING:
{chr(10).join([f"{target}" for target in drop_targets]) if drop_targets else "None"}

AVAILABLE ATTRIBUTES (with zero text overlap):
{chr(10).join(available_zero_score)}

TASK: Among the available attributes, select the attributes that are most irrelevant in nature, sensitivity, or context to the selected abstract/drop attributes.

RULES:
1. Choose attributes that have the most irrelevant privacy sensitivity levels
2. Consider the most irrelevant semantic similarity
3. Consider the most irrelevant contextual similarity
4. Do not select attributes that include or involve any of the abstract/drop attributes
5. Choose 1-2 attributes total (if possible)

Output format: Return only the selected attribute names, one per line, without any additional text or formatting.

Selected attributes to keep:"""

        # Get LLM response
        response = self.model.interact(prompt, temperature=0)
        
        # Parse the response to extract selected attributes
        selected_keep_attrs = []
        for line in response.strip().split('\n'):
            attr_name = line.strip().strip('- ').strip()
            if attr_name:
                selected_keep_attrs.append(attr_name)
        
        # Match selected attributes with available zero-score attributes
        matched_grouped = []
        matched_individual = {}
        
        for selected_attr in selected_keep_attrs:
            # Try to match with zero-score attributes
            matched = False
            for attr_name, attr_info in zero_score_attrs.items():
                original_key = attr_info['original_key']
                if selected_attr.lower() in original_key.lower() or original_key.lower() in selected_attr.lower():
                    if attr_info['type'] == 'group':
                        matched_grouped.append(original_key)
                    else:
                        matched_individual[original_key] = individual_attrs[original_key]
                    matched = True
                    break
        
        # Step 5: Check for overlap with abstract_attrs and drop_attrs
        final_selected = []
        selected_keys = set()
        
        # Extract keys from abstract and drop attributes for overlap checking
        abstract_keys = set()
        drop_keys = set()
        
        if abstract_attrs['selected'] is not None:
            if abstract_attrs['group']:
                for group in abstract_attrs['selected']:
                    abstract_keys.update(group['attrs'].keys())
            else:
                abstract_keys.update(abstract_attrs['selected'].keys())
        
        if drop_attrs['selected'] is not None:
            if drop_attrs['group']:
                for group in drop_attrs['selected']:
                    drop_keys.update(group['attrs'].keys())
            else:
                drop_keys.update(drop_attrs['selected'].keys())
        
        # Filter out overlapping attributes
        for group_name in matched_grouped:
            if group_name not in abstract_keys and group_name not in drop_keys:
                final_selected.append(('group', group_name))
                selected_keys.add(group_name)
        
        for key in matched_individual:
            if key not in abstract_keys and key not in drop_keys:
                final_selected.append(('individual', key))
                selected_keys.add(key)
        
        # Step 6: Create the result structure
        if not final_selected:
            return {'selected': None, 'group': False}
        
        # Determine if we should return grouped or individual attributes
        group_count = sum(1 for attr_type, _ in final_selected if attr_type == 'group')
        
        result = {
            'group': True if random.random() < 0.5 and group_count > 0 else False,
            'selected': None
        }
        
        if result['group']:
            # Return grouped attributes
            selected_groups = []
            for attr_type, attr_name in final_selected:
                if attr_type == 'group':
                    selected_groups.append({
                        'attrs': grouped_attrs[attr_name],
                        'group_name': attr_name
                    })
            result['selected'] = selected_groups
        else:
            # Return individual attributes
            selected_individual = {}
            for attr_type, attr_name in final_selected:
                if attr_type == 'individual':
                    selected_individual[attr_name] = individual_attrs[attr_name]
            result['selected'] = selected_individual
            result['group'] = False
        
        return result
    
    def _find_closest_attributes(self, grouped_attrs: dict, individual_attrs: dict, 
                                abstract_attrs: dict, drop_attrs: dict) -> dict:
        """
        Use LLM to find the closest attributes to the selected abstract and drop attributes.
        
        Args:
            grouped_attrs (dict): Available grouped attributes
            individual_attrs (dict): Available individual attributes
            abstract_attrs (dict): Selected attributes for abstraction
            drop_attrs (dict): Selected attributes for dropping
            model_name (str): LLM model to use
            
        Returns:
            dict: Selected keep attributes that are closest to abstract/drop attributes
        """
        # Prepare the selected attributes for the prompt
        abstract_targets = []
        drop_targets = []
        
        # Extract abstract targets
        if abstract_attrs['selected'] is not None:
            if abstract_attrs['group']:
                for group in abstract_attrs['selected']:
                    abstract_targets.append(f"- {group['group_name']}")
                    # Append individual attributes within the group
                    for key, val in group['attrs'].items():
                        abstract_targets.append(f"  - {key}: {val}")
            else:
                for key, val in abstract_attrs['selected'].items():
                    abstract_targets.append(f"- {key}: {val}")
        
        # Extract drop targets
        if drop_attrs['selected'] is not None:
            if drop_attrs['group']:
                for group in drop_attrs['selected']:
                    drop_targets.append(f"- {group['group_name']}")
                    # Append individual attributes within the group
                    for key, val in group['attrs'].items():
                        drop_targets.append(f"  - {key}: {val}")
            else:
                for key, val in drop_attrs['selected'].items():
                    drop_targets.append(f"- {key}: {val}")
        
        # Prepare available attributes for selection
        available_grouped = list(grouped_attrs.keys())
        available_individual = []
        for key, val in individual_attrs.items():
            available_individual.append(f"- {key}: {val}")
        
        # Build the prompt
        prompt = f"""You are an **Attribute Similarity Analyzer**.

I have selected some attributes for sanitization (abstraction and dropping). I need you to find the most similar attributes from the available pool that should be kept (not sanitized).

SELECTED ATTRIBUTES FOR ABSTRACTION:
{chr(10).join([f"{target}" for target in abstract_targets]) if abstract_targets else "None"}

SELECTED ATTRIBUTES FOR DROPPING:
{chr(10).join([f"{target}" for target in drop_targets]) if drop_targets else "None"}

AVAILABLE INDIVIDUAL ATTRIBUTES:
{chr(10).join(available_individual)}

TASK: Select the attributes that are most similar in nature, sensitivity, or context to the selected abstract/drop attributes, but they should not include or involve any of the selected abstract/drop attributes. These will be kept as-is.

RULES:
1. Choose attributes that have similar privacy sensitivity levels
2. Consider semantic similarity
3. Consider contextual similarity
4. Consider data type similarity
5. Do not select attributes that include or involve any of the selected abstract/drop attributes
6. Choose 2-4 attributes total

Output format: Return only the selected attribute names, one per line, without any additional text or formatting.

Selected attributes to keep:"""

        # Get LLM response
        response = self.model.interact(prompt, temperature=0.3)
        
        # Parse the response to extract selected attributes
        selected_keep_attrs = []
        for line in response.strip().split('\n'):
            attr_name = line.strip().strip('- ').strip()
            if attr_name:
                selected_keep_attrs.append(attr_name)
        
        # Match selected attributes with available attributes
        matched_grouped = []
        matched_individual = {}
        
        for selected_attr in selected_keep_attrs:
            # Try to match with grouped attributes first
            matched = False
            for group_name in available_grouped:
                if selected_attr.lower() in group_name.lower() or group_name.lower() in selected_attr.lower():
                    matched_grouped.append(group_name)
                    matched = True
                    break
            
            # If not matched with grouped, try individual attributes
            if not matched:
                for ind_attr, ind_val in individual_attrs.items():
                    if selected_attr.lower() in ind_attr.lower() or ind_attr.lower() in selected_attr.lower():
                        matched_individual[ind_attr] = ind_val
                        matched = True
                        break
        
        # Create the result structure
        result = {
            'group': len(matched_grouped) > len(matched_individual),
            'selected': None
        }
        
        if result['group'] and matched_grouped:
            # Return grouped attributes
            selected_groups = []
            for group_name in matched_grouped:
                selected_groups.append({
                    'attrs': grouped_attrs[group_name],
                    'group_name': group_name
                })
            result['selected'] = selected_groups
        elif matched_individual:
            # Return individual attributes
            result['selected'] = matched_individual
            result['group'] = False

        return result

def main(args):
    sanitizer = PrivasisSanitizer(args)
    try:
        sanitizer.run()
        # sanitizer.save_clean_csv()
    finally:
        # Close the model client to release HTTP connection pools
        # This prevents the script from hanging after all workers are done
        if hasattr(sanitizer, 'model') and hasattr(sanitizer.model, 'close'):
            sanitizer.model.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--privasis-data-id',
                        help="Run ID that was used for the generation phase, the Abstractor class will try to load the file under data/outputs/privasis/ with this ID.",
                        type=str,
                        default="mark07")
    parser.add_argument('--run-id',
                        help="Run ID, which will also be used as the output file name under data/abstracted_privasis/",
                        type=str,
                        default="v0")
    parser.add_argument('--retry-limit',
                        help="The number of times to retry the abstraction if it fails.",
                        type=int,
                        default=1)
    parser.add_argument('--attr-selection-weighting', type=str, choices=['uniform', 'sensitivity'], default='sensitivity', help="The weighting strategy for selecting attributes for abstraction")
    parser.add_argument('--print', action='store_true', help="Print the abstracted outputs")
    parser.add_argument('--sanitization-model', type=str, default='hf/openai/gpt-oss-120b', help="The name of the sanitization model to use")
    
    # vLLM server arguments (for parallel processing)
    parser.add_argument('--vllm-server-url', type=str, default=None, 
                       help="URL of vLLM server (e.g., http://localhost:8000/v1). "
                            "If provided, connects to external vLLM server instead of loading model locally.")
    parser.add_argument('--num-workers', type=int, default=1,
                       help="Number of parallel worker threads for sanitization (default: 1). "
                            "Only effective when using --vllm-server-url.")
    
    args = parser.parse_args()
    main(args)
