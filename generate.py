import os
import json
import random
import argparse
import threading
import logging
import functools
import traceback
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple, List, Optional
from openai import OpenAI
from tqdm import tqdm
import tiktoken
import pandas as pd

import colorful as cf
cf.use_true_colors()
cf.use_style('monokai')
from rich import print
from rich.panel import Panel
from rich import box

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

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
class PrivasisError(Exception):
    """Base exception for Privasis generation errors."""
    pass


class ProfileGenerationError(PrivasisError):
    """Error during profile generation."""
    pass


class RecordGenerationError(PrivasisError):
    """Error during record generation."""
    pass


class AttributeProcessingError(PrivasisError):
    """Error during attribute processing."""
    pass


class ModelInteractionError(PrivasisError):
    """Error during model interaction."""
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
from utils.load_data import (
    load_jsonl,
    load_names,
    load_countries,
    load_religions,
    load_profile
)
from utils.misc import (
    read_logs,
    generate_random_age,
    generate_random_date,
    generate_random_marital_status,
    generate_random_income_class,
    generate_random_from_list,
    get_dict_basic_event_attrs,
    get_text_hash,
    capture_and_parse_ordered_list,
    compute_embedding_vendi,
)
from templates.schemas import (
    ProfileAndEvents,
    GroupedAttributes
)

PROJECT_HOME = Path(__file__).parent.resolve()
OUTPUT_DIR = os.path.join(PROJECT_HOME, 'outputs', 'privasis')

class PrivasisGenerator:
    """
    The main class for generating Privasis, a million-scale privacy dataset, with step-by-step contextualization.
    run() method is the main method to generate the dataset, which generates profiles, situations, and records.
    We incorporate metropolis-hastings sampling to generate the dataset.
    generate_privasis_data() method is the core method to generate a single data point.
    The metropolis-hastings sampling is implemented in the generate_privasis_data() method.

    - Hierarchical prior: p(z) generates z=(persona, record-type, social-context, format).
        - p(z)=p(persona)p(record-type|persona)p(social-context|persona,record-type)p(format|record-type,social-context).
    - Renderer (independence proposal): q_0(record|z) gives you a plausible but coarse first draft.
    - Local MH moves: q_edit(record'|record) + acceptance step sharpen the draft, adding the specificity and diversity your corpus lacks.
    The renderer is simply an informed initialization and a smart proposal in the language of Metropolis-Hastings. Starting the chain with that structured draft is not just reasonable—it's the standard way to make MCMC mix fast in high-dimensional, structured generative problems.
    """
    def __init__(self, args, online=False) -> None:
        self.args = args
        self.flags = args.flags
        self.online = online
        
        # vLLM configuration
        self.vllm_config = {
            'num_gpus': getattr(args, 'num_gpus', 1),
            'gpu_memory_utilization': getattr(args, 'gpu_memory_utilization', 0.9),
            'max_model_len': getattr(args, 'max_model_len', None),
        }
        
        # vLLM server configuration (for parallel processing)
        self.vllm_server_url = getattr(args, 'vllm_server_url', None)
        self.num_workers = getattr(args, 'num_workers', 1)

        self.name_db = load_names()
        self.country_list = load_countries()
        self.religion_list = load_religions()
        self.seeds = load_jsonl(args.seeds_path) if args.seeds_path else [self.generate_seed() for _ in range(args.n_seeds)]
        self.num_events = args.num_events
        self.profile_template_events = load_profile(os.path.join(PROJECT_HOME, "templates", args.profile_template_events))
        
        self.event_dict = self.load_and_transform_json(os.path.join(PROJECT_HOME, args.event_dict_path))
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "errors"), exist_ok=True)
        self.output_file = os.path.join(OUTPUT_DIR, f"{args.run_id}.jsonl")
        self.error_file = os.path.join(OUTPUT_DIR, "errors", f"{args.run_id}_errors.jsonl")
        self.tolerance = args.tolerance
        
        # Initialize history of embeddings with a maximum size
        self.max_embeddings = 1024  # Maximum number of embeddings to keep
        self.previous_embeddings = []
        self.current_score = 0.0  # Track the current state's score
        
        # Thread-safe locks for parallel processing
        self._embeddings_lock = threading.Lock()
        self._file_lock = threading.Lock()
        self._score_lock = threading.Lock()
        
        # Statistics counters (thread-safe)
        self._stats_lock = threading.Lock()
        self._success_count = 0
        self._failure_count = 0
        
        # Pre-load the model once at initialization
        self.model_name = args.generator_model
        self.supports_structured_output = self.model_name in ["gpt-4.1-nonasync"]
        self.model = load_model(
            self.model_name,
            num_gpus=self.vllm_config['num_gpus'],
            gpu_memory_utilization=self.vllm_config['gpu_memory_utilization'],
            max_model_len=self.vllm_config['max_model_len'],
            vllm_server_url=self.vllm_server_url,
        )
        
        # Embedding model configuration
        self.embedding_model = getattr(args, 'embedding_model', 'text-embedding-3-small')
        self.embedding_device = getattr(args, 'embedding_device', 'cuda')
        
        # Cache for Huggingface embedding models (lazy loaded)
        self._hf_embedding_model = None
        self._hf_embedding_tokenizer = None
        self._hf_embedding_model_name = None
        self._hf_model_lock = threading.Lock()
        
        # Cache for OpenAI embedding client (lazy loaded, thread-safe)
        self._openai_client = None
        self._openai_client_lock = threading.Lock()
        
        # Load existing embeddings if available
        if os.path.exists(self.output_file):
            prior_results = pd.read_json(self.output_file, lines=True)
            for idx, row in tqdm(prior_results.iterrows(), desc="Loading embeddings", total=len(prior_results)):
                embedding, _ = self.get_embeddings(row['record']['text'])
                self.previous_embeddings.append(embedding)
                # Keep only the most recent embeddings
                if len(self.previous_embeddings) > self.max_embeddings:
                    self.previous_embeddings = self.previous_embeddings[-self.max_embeddings:]

    def generate_seed(self) -> dict:
        seed = {}
        row = self.name_db.sample(1).to_dict('records')[0]
        name = row['name']
        sex = random.choices(["female", "male"], weights=[row['F'], row['M']])[0]

        seed['First Name'] = name
        seed['Date'] = generate_random_date(2000, 2023)
        seed['Age'] = generate_random_age()
        seed['Marital Status'] = generate_random_marital_status()
        seed['Income Class'] = generate_random_income_class()
        seed["Morality level"] = random.choices(
            ["Unethical", "Questionable Morality", "Morally Average", "Ethical", "Exemplary Morality"],
            weights=[0.1, 0.2, 0.3, 0.2, 0.1]
        )[0]

        output_dict = get_dict_basic_event_attrs()
        basic_vals = output_dict['basic_vals']
        seed['Native Language'] = generate_random_from_list(basic_vals["NATIVE_LANGUAGE"]["literal_value"])
        seed['Blood Type'] = generate_random_from_list(basic_vals["BLOOD_TYPE"]["literal_value"])

        if "other_characteristics" in self.flags:
            seed['Sex at Birth'] = sex
            seed['Country of Birth'] = random.choice(self.country_list)
            seed["Disability Status"] = random.choice(["Yes", "No"])

        return seed

    def load_and_transform_json(self,file_path):
        # Load the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # Transform the list into a dictionary
        transformed_dict = {}
        for entry in data:
            event_name = entry["event"]
            attributes = {
                "pivotal_attributes": entry["pivotal_attributes"],
                #"non_pivotal_attributes": entry["non_pivotal_attributes"]
            }
            transformed_dict[event_name] = attributes
        
        return transformed_dict

    def generate_profile_w_event(self, seed: dict) -> Tuple[dict, bool]:
        """
        Generate a profile with a set of events.

        Args:
            seed (dict): seed values for the profile generation

        Returns:
            Tuple[dict, bool]: 
             - profile and event information
             - error flag
        """
        # reformat events
        def reformat_events(event_attrs):
            out = []
            for rec in event_attrs:
                new = {'event': rec.get('event')}
                attrs_list = rec.get('attributes', [])
                attrs_dict = {}
                # Move other top-level keys (like 'date') into attributes
                for k, v in rec.items():
                    if k not in ('event', 'attributes'):
                        attrs_dict[k] = v
                # Convert list of name/value pairs to dict
                for item in attrs_list:
                    name = item.get('name')
                    value = item.get('value')
                    if name in ('destinations', 'content_type') and isinstance(value, str):
                        parts = [p.strip() for p in value.split(',') if p.strip()]
                        attrs_dict[name] = parts
                    else:
                        attrs_dict[name] = value
                new['attributes'] = attrs_dict
                out.append(new)
            return out

        output_dict = get_dict_basic_event_attrs()
        basic_vals = output_dict['basic_vals']
        options = {
            'Ethnicity': basic_vals["ETHNICITY"]["literal_value"],
            'Sex': basic_vals["SEX"]["literal_value"]
        }

        # Select random events from the event_dict
        selected_events = random.sample(list(self.event_dict.keys()), self.num_events)
        
        events_string = ""
        for event in selected_events:
            attrs_piv = list(self.event_dict[event]['pivotal_attributes'].keys())
            attr_string = ", ".join(attrs_piv)
            # attrs_nonpiv = list(self.event_dict[event]['non_pivotal_attributes'].keys())
            # attr_string += ", ".join(attrs_nonpiv)
            events_string += f"- Event: {event}\n  Attributes (choose as many as necessary): {attr_string}\n"

        formatted_seed = "\n".join([f"- {k}: {v}" for k, v in seed.items()]).strip()
        seeded_profile = self.profile_template_events.format(formatted_seed, options['Sex'], options['Ethnicity'], events_string)

        if self.supports_structured_output:
            completed_profile_detailed = self.model.interact(prompt=seeded_profile, max_tokens=4096, response_format=ProfileAndEvents)
        else:  # for models without response_format
            completed_profile_detailed = self.model.interact(prompt=seeded_profile, max_tokens=4096)

        if isinstance(completed_profile_detailed, dict) and "error" in completed_profile_detailed:
            print(Panel(f"Error in populating profiles:\n{completed_profile_detailed['failures']}", box=box.SIMPLE_HEAD, title="Error", expand=False, style="red"))
            return {"error": "[profile-gen] Model returned error during profile generation"}, True

        try:
            parsed_profile = completed_profile_detailed.dict()
        except AttributeError:
            # Model returned a string instead of a Pydantic object, try to parse as JSON
            completed_profile_detailed = completed_profile_detailed.replace("```json", "").replace("```", "").strip()
            try:
                parsed_profile = json.loads(completed_profile_detailed)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse profile JSON: {e}")
                return {"error": f"[profile-parse] Invalid JSON from model: {e}"}, True

        # reformat profile and events
        if self.supports_structured_output:
            reformatted_profile = {attr["name"]: attr["value"] for attr in parsed_profile["profile"]}
            reformatted_events = [{
                "event": event["event"],
                "attributes": {attr["name"]: attr["value"] for attr in event["attributes"]}
            } for event in parsed_profile["events"]]
        else: # for models without response_format
            reformatted_profile = parsed_profile['profile']
            reformatted_events = reformat_events(parsed_profile["events"])

        return {
            "initial_demographic_seed": seed,
            "events_string": events_string,
            "generated_profile_string": str(parsed_profile),
            "profile": reformatted_profile,
            "event_list": reformatted_events,
            # "profile_and_events_text": profile_text,
            "id": get_text_hash(str(parsed_profile)),
        }, False

    def select_attributes(self, attr_dict: dict) -> Tuple[str, any]:
        """
        Select an attribute from the event attributes dictionary.
        
        Args:
            attr_dict: Dictionary containing 'attributes' key with attribute data
            
        Returns:
            Tuple of (attribute_type, attribute_value)
            
        Raises:
            ValueError: If no valid attributes are available after filtering
        """
        attr_keys = list(attr_dict['attributes'].keys())
        attr_keys_to_exclude = ['details', 'event_type', 'entry']
        for attr_key_to_exclude in attr_keys_to_exclude:
            if attr_key_to_exclude in attr_keys:
                attr_keys.remove(attr_key_to_exclude)
        
        if not attr_keys:
            raise ValueError(f"No valid attributes found after filtering. Original keys: {list(attr_dict['attributes'].keys())}")
        
        attr_type = random.choice(attr_keys)
        attr_value = attr_dict['attributes'][attr_type]
        return (attr_type, attr_value)

    @retry(max_attempts=3, delay=1.0, exceptions=(json.JSONDecodeError, ModelInteractionError))
    def get_included_profile_attributes(self, record, profile) -> dict:
        """
        Extract profile attributes that are explicitly mentioned in the record.
        
        Args:
            record: The record text to analyze
            profile: The profile template for reference
            
        Returns:
            Dictionary of included profile attributes
            
        Raises:
            json.JSONDecodeError: If response cannot be parsed as JSON
            ModelInteractionError: If model interaction fails
        """
        prompt = f"{record}\n\nQuestion: Extract all profile attributes and their values that are explicitly mentioned in the record above. Only include attributes that are directly stated in the record text, and use the exact values as they appear in the record. Do not include any attributes that are not explicitly mentioned in the record. Output in a JSON format and do not include any additional comments.\n\nProfile template (for reference only):\n{json.dumps(profile, indent=4)}"
        
        try:
            if self.supports_structured_output:
                included_profile_str = self.model.interact(prompt=prompt, json_mode=True, max_tokens=5024)
            else:  # for models without response_format
                included_profile_str = self.model.interact(prompt=prompt, max_tokens=5024)
                included_profile_str = included_profile_str.replace("```json", "").replace("```", "").strip()
        except Exception as e:
            raise ModelInteractionError(f"Model interaction failed: {e}") from e
        
        return json.loads(included_profile_str)

    def group_attributes(self, attrs: dict, event_description: str, record_text: str) -> Tuple[dict, bool]:
        """
        Organize the flat event attributes JSON into nested groups of attributes.
        """

        # Create index map (idx -> attribute key)
        idx_map = {i: {k: v} for i, (k, v) in enumerate(attrs.items())}
        prompt = f"""Event description: {event_description}

Indexed event attributes:
{json.dumps(idx_map, indent=4)}

**Task**: Group the above attributes into meaningful clusters for privacy analysis and data protection purposes.

**Instructions**:
1. **Create semantic clusters**: Group attributes that are conceptually related or serve similar purposes
2. **Consider privacy implications**: Focus on how attributes could be used together for privacy analysis
3. **Use descriptive cluster names**: Create clear, specific labels that indicate the cluster's purpose
4. **Allow overlap**: Attributes can belong to multiple clusters if they serve different purposes in different contexts
5. **Aim for 3-8 clusters**: Create enough clusters to be meaningful but not so many that they become fragmented
6. **Be specific**: Use very specific and descriptive names for the clusters. For example, if the attributes are about a specific person, the cluster name should include the person's name.

**Output Format**: A json object with the following format:
{{
    "clusters": [
        {{
            "cluster_name": "Cluster 1",
            "attributes": [0, 1, 2]
        }},
        {{
            "cluster_name": "Cluster 2",
            "attributes": [1, 4, 5]
        }}
    ]
}}

- Each cluster should have a very specific and descriptive name that clearly indicates the cluster's purpose
- Each cluster should have a list of attribute indices (0, 1, 2, etc.) that belong to that cluster

Focus on creating clusters that would be useful for privacy analysis, data anonymization, or understanding what types of sensitive information are present in the event. Do not include any additional comments."""

        if self.supports_structured_output:
            grouped_event_attributes = self.model.interact(prompt=prompt, max_tokens=4096, response_format=GroupedAttributes) 
        else:  # for models without response_format
            grouped_event_attributes = self.model.interact(prompt=prompt, max_tokens=4096)
            grouped_event_attributes = grouped_event_attributes.replace("```json", "").replace("```", "").strip()
            try:
                grouped_event_attributes = json.loads(grouped_event_attributes)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse grouped attributes JSON: {e}")
                return {'error': f'[attr-group] Invalid JSON from model: {e}', 'grouped_event_attrs_str': grouped_event_attributes}, True
        if isinstance(grouped_event_attributes, dict) and "error" in grouped_event_attributes:
            print(Panel(f"Error in grouping attributes:\n{grouped_event_attributes['failures']}", box=box.SIMPLE_HEAD, title="Error", expand=False, style="red"))
            return {'error': '[attr-group] Model returned error response', 'grouped_event_attrs_str': grouped_event_attributes['failures']}, True

        # Parse index clusters
        parsed_grouped_event_attributes = {}
        if self.supports_structured_output:
            for cluster in grouped_event_attributes.clusters:
                parsed_attrs = {}
                for i in cluster.attributes:
                    parsed_attrs.update(idx_map[i])
                parsed_grouped_event_attributes[cluster.cluster_name] = parsed_attrs
        else: # for models without response_format
            for cluster in grouped_event_attributes['clusters']:
                parsed_attrs = {}
                for i in cluster['attributes']:
                    parsed_attrs.update(idx_map[i])
                parsed_grouped_event_attributes[cluster['cluster_name']] = parsed_attrs

        return parsed_grouped_event_attributes, False

    def expand_attributes(self, profile: dict, situation: dict, record: dict) -> dict:
        """
        Annotate the attributes in the record draft based on the profile and the situation.
        
        Args:
            profile: Profile information dictionary
            situation: Situation/context dictionary
            record: Record dictionary containing 'text' key
            
        Returns:
            Dictionary with 'attributes' and 'grouped_attributes', or 'error_message' on failure
        """
        # Get included profile attributes with retry logic (handled by decorator)
        try:
            included_profile_attrs = self.get_included_profile_attributes(record['text'], profile['profile'])
        except (json.JSONDecodeError, ModelInteractionError) as e:
            logger.error(f"Failed to get included profile attributes after retries: {e}")
            return {"error_message": f"[attr-extract] Failed to get profile attributes: {e}"}
        
        name = profile['profile']['first_name']
        event = profile['event_list'][0]

        prompt = f"""Extract all important or sensitive attributes from the following record that relate to {name} or the described event.

**Record:**
{record['text']}

**Instructions:**
1. Identify all attributes that contain personal, sensitive, or important information about {name} or the event
2. Focus on attributes that could be used for privacy analysis or data protection
3. Extract the exact values as they appear in the record
4. Use descriptive, specific attribute names (e.g., "medical_condition" not just "condition")
5. Make the values short and concise, but specific, and do not use generic values like "yes" or "no" or "true" or "false" unless they are specific to the attribute.

**What to exclude:**
- Attributes not present in the record
- Binary attributes (yes/no, true/false) unless they contain specific details
- Generic or vague attributes without concrete values
- Attributes that are too long
- Duplicate information

**Output format:** JSON with attribute names as keys and their exact values as values. Do not use nested structures. Do not include any additional comments."""

        # Extract attributes from record
        attr_str = None
        try:
            if self.supports_structured_output:
                attr_str = self.model.interact(prompt=prompt, json_mode=True, max_tokens=7000)
            else:  # for models without response_format
                attr_str = self.model.interact(prompt=prompt, max_tokens=7000)
                attr_str = attr_str.replace("```json", "").replace("```", "").strip()
            attrs = json.loads(attr_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse attributes JSON: {e}\nResponse: {attr_str[:500]}...")
            print(Panel(f"Error in getting attributes:\n{attr_str}", box=box.SIMPLE_HEAD, title="Error", expand=False, style="red"))
            return {"error_message": f"[attr-extract] Invalid JSON from model: {e}"}
        except Exception as e:
            logger.error(f"Unexpected error parsing attributes during post-processing: {e}")
            return {"error_message": f"[attr-extract] Unexpected error: {e}"}

        # Organize attributes into profile and event categories
        organize_prompt = f"Profile:\n{json.dumps(included_profile_attrs, indent=4)}\n\nEvent attributes:\n{json.dumps(attrs, indent=4)}\n\nOrganize the two JSONs and output one JSON with keys 'profile' and 'event' according to the above. Do not have duplicate values across 'profile' and 'event'. Do not include any additional comments."
        
        try:
            if self.supports_structured_output:
                organized_attr_str = self.model.interact(prompt=organize_prompt, json_mode=True, max_tokens=7000)
            else:  # for models without response_format
                organized_attr_str = self.model.interact(prompt=organize_prompt, max_tokens=7000)
                organized_attr_str = organized_attr_str.replace("```json", "").replace("```", "").strip()
            organized_attrs = json.loads(organized_attr_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse organized attributes JSON: {e}")
            print(Panel(f"Error in getting organized attributes:\n{organized_attr_str}", box=box.SIMPLE_HEAD, title="Error", expand=False, style="red"))
            return {"error_message": f"[attr-organize] Invalid JSON from model: {e}"}
        except Exception as e:
            logger.error(f"Unexpected error organizing attributes: {e}")
            return {"error_message": f"[attr-organize] Unexpected error: {e}"}

        # Add full name if not present
        if "full_name" not in organized_attrs.get('profile', {}) and "first_name" in profile['profile'] and "last_name" in profile['profile']:
            if 'profile' not in organized_attrs:
                organized_attrs['profile'] = {}
            organized_attrs['profile']['full_name'] = f"{profile['profile']['first_name']} {profile['profile']['last_name']}"

        # Validate that event attributes are not empty
        # Empty event attributes will cause sanitization to fail downstream
        if not organized_attrs.get('event', {}):
            logger.warning(f"No event attributes extracted from record. Record text preview: {record['text'][:200]}...")
            return {"error_message": "[attr-validate] No event attributes extracted - record may not contain relevant event information"}

        # Group attributes
        try:
            combined_attrs = {**organized_attrs.get('profile', {}), **organized_attrs.get('event', {})}
            grouped_attrs, error_flag = self.group_attributes(
                attrs=combined_attrs, 
                event_description=event['event'], 
                record_text=record['text']
            )
            if error_flag:
                logger.warning(f"Error flag set in group_attributes: {grouped_attrs}")
        except (KeyError, TypeError) as e:
            logger.error(f"Error in getting grouped attributes: {e}\nOrganized attrs: {organized_attrs}")
            print(Panel(f"Error in getting grouped attributes:\n{organized_attrs}", box=box.SIMPLE_HEAD, title="Error", expand=False, style="red"))
            return {"error_message": f"[attr-group] Failed to group attributes: {e}"}

        return {'attributes': organized_attrs, 'grouped_attributes': grouped_attrs}

    def sample_record_types(self, target_attr: tuple, profile_info: dict, top_n=10) -> List[str]:
        """
        Sample 'n' record types based on the target attribute and profile information.
        """
        profile = profile_info['profile']
        if len(profile_info['event_list']) == 1:
            random_event_idx = 0
        else:
            random_event_idx = random.randint(1, len(profile_info['event_list'])-1) # get a random number from 1 to len(event_list) to avoid the date bias
        event_attrs = profile_info['event_list'][random_event_idx]
        attr_type, attr_value = target_attr
        if "first_name" in profile:
            name = profile['first_name']
        elif "name" in profile:
            name = profile['name']
        elif "full_name" in profile:
            name = profile['full_name']
        else:
            return None
        record_formality = random.choice(['formal medium', 'formal medium', 'formal medium', 'informal medium'])

        prompt = f"""Profile:
{json.dumps(profile, indent=4)}

Event:
{json.dumps(event_attrs, indent=4)}

Generate a diverse and realistic list of types of {record_formality} that contains {name}'s private information '{attr_type}', which is '{attr_value}'. 

Requirements:
1. Each record type must specify the exact source/organization (e.g., "Reddit post from r/relationships", "Patient record from Mayo Clinic", "Tax document from IRS", "Employment record from Google")
2. Include a mix of different contexts (personal, professional, medical, legal, financial, etc.)
3. Consider both digital and physical record types
4. Include both common and unusual/unique record types
5. Ensure all types are textual (no images/videos)
6. Each record type should be specific and detailed, not generic

Output the list as an ordered list, with each type being specific and detailed. Do not include any additional comments."""
        model = self.model
        record_types_str = model.interact(prompt=prompt, temperature=1.2, max_tokens=2048)
        record_types = capture_and_parse_ordered_list(record_types_str)
        record_types = [record_type.strip() for record_type in record_types][:top_n]
        if record_types == []:
            record_types_str = model.interact(prompt=prompt, temperature=1.3, max_tokens=2048)
            record_types = capture_and_parse_ordered_list(record_types_str)
            record_types = [record_type.strip() for record_type in record_types][:top_n]

        if not record_types:
            logger.warning(f"No record types generated for attribute {attr_type}={attr_value}")
            return None
        
        return random.choice(record_types)

    def generate_situation(self, attr_to_share: tuple, profile_info: dict, record_type: str) -> str:
        """
        Generate a situation based on the target attribute, profile information, and the record type.
        """
        model = self.model
        profile = profile_info['profile']
        event_attrs = profile_info['event_list'][0]

        prompt = f"""Profile:
{json.dumps(profile, indent=4)}

Event:
{json.dumps(event_attrs, indent=4)}

Generate five creative and specific contexts for the '{record_type}' that contains {profile['first_name']}'s '{attr_to_share[0]}', which is '{attr_to_share[1]}'.

Requirements: Each context should be specific, detailed, realistic and plausible. Include a diverse range of emotional contexts. Consider unique different scenarios, and be detailed enough to understand the situation.

Output the five contexts in an ordered list with each item in plain text. Each context should be specific and detailed, providing a clear situation without generating the actual record. Do not include any additional comments."""
        situations = model.interact(prompt=prompt, temperature=1, max_tokens=4096)
        situations_list = capture_and_parse_ordered_list(situations)
        if situations_list == []:
            situations = model.interact(prompt=prompt, temperature=1.2, max_tokens=4096)
            situations_list = capture_and_parse_ordered_list(situations)

        if not situations_list:
            logger.warning(f"No situations generated for record_type={record_type}")
            return {'error_message': "[situation-gen] Model returned empty list"}
        
        situation = random.choice(situations_list)
        situation_id = get_text_hash(situation)

        return {'text': situation, 'prompt': prompt, 'id': situation_id}

    def sample_format(self, record_type: str, situation: str) -> dict:
        """
        Generate a structure for the record based on the record type and situation.
        """
        model = self.model

        prompt = f"""Record type: {record_type}
        
Situation: {situation}

Based on the situation described above, outline what the structure should look like for '{record_type}'. Describe ten diverse and realistic possible structures and their tone for the '{record_type}' written in plain text in an ordered list. Do not include any values, just a plain description of the structure and tone. Tone should be realistic and diverse, not too cheerful."""
        formats_str = model.interact(prompt=prompt, max_tokens=4096)
        formats = capture_and_parse_ordered_list(formats_str)
        if formats == []:
            formats_str = model.interact(prompt=prompt, temperature=1.2, max_tokens=4096)
            formats = capture_and_parse_ordered_list(formats_str)

        if not formats:
            logger.warning(f"No formats generated for record_type={record_type}")
            return {'error_message': "[format-gen] Model returned empty list"}

        return random.choice(formats)

    def draft_record(self, prior: dict) -> Tuple[dict, bool]:
        """
        Draft a record based on the hierarchical prior.
        This is the independence proposal q_0(record|z) that gives a plausible but coarse first draft.
        """
        profile = prior['profile']
        situation_info = prior['social_context']
        record_type = prior['record_type']
        form = prior['format']
        
        event_info = {**profile['event_list'][0]}
        if 'event' in event_info:
            event_info['event title'] = event_info.pop('event')
            event_info['event attributes'] = event_info.pop('attributes')

        prompt = f"""{profile['profile']['first_name']}'s profile:
{json.dumps(profile['profile'], indent=4)}

Event:
{json.dumps(event_info, indent=4)}

Situation:
{situation_info['text']}

---

Now generate a realistic, detailed and creative '{record_type}' in English according to {profile['profile']['first_name']}'s profile, event, and situation above. Follow these guidelines:

1. Use Profile Information:
   - Incorporate only the relevant attributes from {profile['profile']['first_name']}'s profile and the event above
   - Ensure all personal details match the profile exactly

2. Add Specific Details:
   - Include exact dates (avoid using 15th, use other random dates)
   - Specify precise locations with addresses or landmarks
   - Add realistic timestamps and durations
   - Include specific measurements, quantities, and numbers
   - Instead of monotonic numbers (e.g., 12345, 9876), use other random numbers
   - Use accurate terminology and jargon

3. Structure and Format:
   - Follow the specified structure and tone: {form}
   - Keep the text dense and information-rich
   - Minimize markdown formatting

Only output the generated '{record_type}' without any additional comments or explanations."""

        model = self.model
        # history = [situation_info['prompt'], situation_info['text']]
        try:
            record = model.interact(prompt, temperature=0.7, max_tokens=10240)
        except Exception as e:
            logger.warning(f"First attempt to draft record failed: {e}. Retrying with adjusted parameters.")
            # Retry with different temperature for Gemini models, or same params for others
            retry_temp = 1.0 if "gemini-" in self.model_name else 0.7
            record = model.interact(prompt, temperature=retry_temp, max_tokens=10240)

        record_id = get_text_hash(record)
        output = {'text': record, 'prompt': prompt, 'type': record_type, 'form': form, 'id': record_id}
        if self.args.print:
            print(Panel(record, box=box.SIMPLE_HEAD, title="Draft Record", expand=False, style="blue"))
        return output, False

    def _load_hf_embedding_model(self, model_name: str):
        """Lazy load Huggingface embedding model and tokenizer (thread-safe).
        
        Args:
            model_name: The Huggingface model name (e.g., "Qwen/Qwen3-Embedding-0.6B")
        """
        # Double-checked locking pattern for thread safety
        if self._hf_embedding_model is None or self._hf_embedding_model_name != model_name:
            with self._hf_model_lock:
                # Check again inside the lock
                if self._hf_embedding_model is None or self._hf_embedding_model_name != model_name:
                    logger.info(f"Loading Huggingface embedding model: {model_name} on device: {self.embedding_device}")
                    self._hf_embedding_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                    
                    use_cpu = self.embedding_device == "cpu"
                    torch_dtype = torch.float32 if use_cpu else torch.float16
                    
                    self._hf_embedding_model = AutoModel.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        torch_dtype=torch_dtype
                    )
                    
                    # Move to specified device
                    if use_cpu:
                        self._hf_embedding_model = self._hf_embedding_model.cpu()
                    elif torch.cuda.is_available():
                        self._hf_embedding_model = self._hf_embedding_model.to(self.embedding_device)
                    
                    self._hf_embedding_model.eval()
                    self._hf_embedding_model_name = model_name
                    logger.info(f"Loaded Huggingface embedding model: {model_name}")
    
    def _get_hf_embeddings(self, texts: List[str], model_name: str) -> List[torch.Tensor]:
        """Get embeddings using a Huggingface model.
        
        Args:
            texts: List of texts to embed
            model_name: The Huggingface model name
            
        Returns:
            List of embedding tensors
        """
        self._load_hf_embedding_model(model_name)
        
        device = next(self._hf_embedding_model.parameters()).device
        embeddings = []
        
        for text in texts:
            # Tokenize with truncation
            inputs = self._hf_embedding_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=8192,
                padding=True
            ).to(device)
            
            with torch.no_grad():
                outputs = self._hf_embedding_model(**inputs)
                # Use last hidden state with mean pooling
                # For Qwen3-Embedding, we use the last_hidden_state
                if hasattr(outputs, 'last_hidden_state'):
                    hidden_states = outputs.last_hidden_state
                    # Mean pooling over non-padding tokens
                    attention_mask = inputs['attention_mask']
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                    embedding = (sum_embeddings / sum_mask).squeeze(0)
                else:
                    # Fallback for models that return embeddings directly
                    embedding = outputs[0].mean(dim=1).squeeze(0)
                
                # Normalize the embedding
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
                embeddings.append(embedding.float())  # Convert to float32 for consistency
        
        return embeddings

    def get_embeddings(self, record: str, new_record: str = None, model: str = None) -> np.ndarray:
        """Get embeddings for two records using OpenAI's API or Huggingface models.
        
        Args:
            record: The original record
            new_record: The new record to compare against
            model: Embedding model to use. If None, uses self.embedding_model. Supports:
                   - OpenAI models: "text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"
                   - Huggingface models: "Qwen/Qwen3-Embedding-4B" or any HF model path
        
        Returns:
            embedding_old: The embedding of the original record (as GPU tensor if available)
            embedding_new: The embedding of the new record (as GPU tensor if available)
        """
        if model is None:
            model = self.embedding_model
            
        texts = [record]
        if new_record is not None:
            texts.append(new_record)
        
        # Check if it's a Huggingface model (contains "/" or known HF model names)
        is_huggingface_model = "/" in model or model.startswith("Qwen")
        
        if is_huggingface_model:
            # Use Huggingface model
            embeddings = self._get_hf_embeddings(texts, model)
            embedding_old = embeddings[0]
            if new_record is not None:
                embedding_new = embeddings[1]
                return embedding_old, embedding_new
            else:
                return embedding_old, None
        else:
            # Use OpenAI API (lazy-init, thread-safe, reused across calls)
            if self._openai_client is None:
                with self._openai_client_lock:
                    if self._openai_client is None:
                        api_key = os.getenv("OPENAI_API_KEY")
                        if not api_key:
                            raise ValueError("OPENAI_API_KEY environment variable not set")
                        self._openai_client = OpenAI(api_key=api_key)
            client = self._openai_client
            
            # Truncate texts to 8192 tokens if needed
            encoding = tiktoken.encoding_for_model(model)
            truncated_texts = []
            for text in texts:
                tokens = encoding.encode(text)
                if len(tokens) > 8192:
                    tokens = tokens[:8192]
                    text = encoding.decode(tokens)
                truncated_texts.append(text)
            
            response = client.embeddings.create(
                model=model,
                input=truncated_texts
            )
            embeddings = [data.embedding for data in response.data]
            
            # Convert to GPU tensors if available
            if torch.cuda.is_available():
                device = torch.device('cuda')
                embedding_old = torch.tensor(embeddings[0], dtype=torch.float32, device=device)
                if new_record is not None:
                    embedding_new = torch.tensor(embeddings[1], dtype=torch.float32, device=device)
                    return embedding_old, embedding_new
                else:
                    return embedding_old, None
            else:
                # Fallback to numpy arrays if GPU not available
                embedding_old = np.array(embeddings[0])
                if new_record is not None:
                    embedding_new = np.array(embeddings[1])
                    return embedding_old, embedding_new
                else:
                    return embedding_old, None

    def calculate_embedding_novelty(self, embedding_old, embedding_new, max_embeddings=300) -> float:
        """
        Calculate the novelty of a new record compared to existing records using Vendi score.
        
        Args:
            embedding_old: The embedding of the original record
            embedding_new: The embedding of the new record
            
        Returns:
            float: The novelty score (higher is more novel)
        """
        # === Embedding diversity using Vendi score ===
        # Calculate Vendi scores for both scenarios (thread-safe access)
        with self._embeddings_lock:
            previous_embeddings_copy = list(self.previous_embeddings)
        
        if previous_embeddings_copy:
            # Calculate Vendi score with current record
            # sample max_embeddings embeddings from the previous embeddings
            if len(previous_embeddings_copy) > max_embeddings:
                sampled_embeddings = random.sample(previous_embeddings_copy, max_embeddings)
            else:
                sampled_embeddings = previous_embeddings_copy

            current_embeddings = sampled_embeddings + [embedding_old]
            # Convert list to numpy array for vendi computation
            if torch.cuda.is_available() and isinstance(current_embeddings[0], torch.Tensor):
                # If using GPU tensors, convert to numpy for vendi computation
                current_embeddings_np = torch.stack(current_embeddings).cpu().numpy()
            else:
                current_embeddings_np = np.array(current_embeddings)
            current_vendi = compute_embedding_vendi(current_embeddings_np)
            # current_vendi = compute_embedding_vendi_original(current_embeddings_np)
            
            # Calculate Vendi score with new record
            new_embeddings = sampled_embeddings + [embedding_new]
            if torch.cuda.is_available() and isinstance(new_embeddings[0], torch.Tensor):
                # If using GPU tensors, convert to numpy for vendi computation
                new_embeddings_np = torch.stack(new_embeddings).cpu().numpy()
            else:
                new_embeddings_np = np.array(new_embeddings)
            new_vendi = compute_embedding_vendi(new_embeddings_np)
            # new_vendi = compute_embedding_vendi_original(new_embeddings_np)
            
            # Novelty is the improvement in Vendi score
            embed_novelty = max(0, new_vendi - current_vendi)
        else:
            # If no previous records, just use direct comparison
            if torch.cuda.is_available() and isinstance(embedding_old, torch.Tensor):
                # GPU tensor cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(
                    embedding_old.unsqueeze(0), 
                    embedding_new.unsqueeze(0)
                ).item()
            else:
                # Numpy cosine similarity
                cos_sim = cosine_similarity([embedding_old], [embedding_new])[0][0]
            embed_novelty = 1.0 - cos_sim
            
        return embed_novelty

    def compare_records(self, record: str, new_record: str, lambda_llm=0.5, lambda_embed=0.5, accept_option="deterministic") -> Tuple[bool, np.ndarray]:
        """
        Compare two records using LLM specificity + Vendi score for diversity.

        Args:
            record (str): The original record
            new_record (str): The new record to compare against
            lambda_llm (float): Weight for LLM specificity score
            lambda_embed (float): Weight for embedding novelty

        Returns:
            Tuple[bool, np.ndarray]: 
                - bool: True if the new record should be kept
                - np.ndarray: The embedding of the new record
        """

        # Randomize order to avoid LLM bias
        if random.random() < 0.5:
            record_one, record_two = record, new_record
            is_new_record_two = True
        else:
            record_one, record_two = new_record, record
            is_new_record_two = False

        prompt = f"""
Compare the following two records and decide which one is more specific and realistic.

**Record 1**: {record_one}

**Record 2**: {record_two}

Respond only with '1' or '2', indicating which record is better."""
        better = self.model.interact(prompt=prompt.strip(), max_tokens=4000).strip()

        # === Embedding diversity using Vendi score ===
        embedding_old, embedding_new = self.get_embeddings(record, new_record)
        embed_novelty = self.calculate_embedding_novelty(embedding_old, embedding_new)

        # === LLM score: 1 if new record wins, else 0 ===
        if ("2" in better and is_new_record_two) or ("1" in better and not is_new_record_two):
            llm_score = 1
            # print(cf.yellow(f"The new record is judged better by LLM."))
        else:
            llm_score = 0
            # print(cf.yellow(f"The existing record is judged better by LLM."))

        # === Combined decision ===
        if accept_option == "deterministic":
            combined_score = lambda_llm * llm_score + lambda_embed * embed_novelty
            keep_new = combined_score > 0.5  # acceptance threshold
            # print(f"Vendi score improvement: {embed_novelty:.3f}, LLM score: {llm_score:.3f}, Combined score: {combined_score:.3f}")
        elif accept_option == "metropolis-hastings":
            # Calculate acceptance probability using Metropolis-Hastings rule (thread-safe)
            with self._embeddings_lock:
                has_previous = bool(self.previous_embeddings)
            with self._score_lock:
                current_score = self.current_score
            
            # Initialize for logging (overwritten in the else branch)
            score_diff = 0.0
            acceptance_prob = 1.0
            
            # For the first record, we always accept
            if not has_previous:
                keep_new = True
                with self._score_lock:
                    self.current_score = lambda_llm * llm_score + lambda_embed * embed_novelty
            else:
                # Calculate scores for both states
                new_score = (lambda_llm * llm_score + lambda_embed * embed_novelty)
                
                # Boost new score when both llm_score is 1 and embed_novelty is positive
                if llm_score == 1 and embed_novelty > 0:
                    # Add a bonus term that increases with embed_novelty
                    new_score += 4.0 * embed_novelty
                
                # Calculate the actual score difference for Metropolis-Hastings
                score_diff = new_score - current_score
                
                # Calculate acceptance probability
                acceptance_prob = min(1.0, np.exp(score_diff))
                # Stochastic acceptance
                keep_new = random.random() < acceptance_prob
                
                # Update current score if we accept the new record (thread-safe)
                if keep_new:
                    with self._score_lock:
                        self.current_score = new_score
            if self.args.print:
                print(f"Vendi score improvement: {embed_novelty:.3f}, LLM score: {llm_score:.3f}, Score difference: {score_diff:.3f}, Acceptance probability: {acceptance_prob:.3f}")

        if self.args.print:
            if keep_new:
                print(cf.green(f"The new record is accepted. Vendi score improvement: {embed_novelty:.3f}, LLM score: {llm_score:.3f}"))
            else:
                print(cf.yellow(f"The new record is rejected. Vendi score improvement: {embed_novelty:.3f}, LLM score: {llm_score:.3f}"))
            print("=" * 40)
            print()

        return keep_new, embedding_new

    def compare_records_llm_only(self, record: str, new_record: str) -> bool:
        """
        Compare two records using only LLM specificity score.

        Args:
            record (str): The original record
            new_record (str): The new record to compare against

        Returns:
            bool: True if the new record should be kept (is more specific/realistic)
        """
        # Randomize order to avoid LLM bias
        if random.random() < 0.5:
            record_one, record_two = record, new_record
            is_new_record_two = True
        else:
            record_one, record_two = new_record, record
            is_new_record_two = False

        prompt = f"""
Compare the following two records and decide which one is more specific and realistic.

**Record 1**: {record_one}

**Record 2**: {record_two}

Respond only with '1' or '2', indicating which record is better."""
        
        better = self.model.interact(prompt=prompt.strip(), max_tokens=10).strip()

        # Determine if the new record is better
        if ("2" in better and is_new_record_two) or ("1" in better and not is_new_record_two):
            keep_new = True
            print(cf.green(f"The new record is judged better by LLM."))
        else:
            keep_new = False
            print(cf.yellow(f"The existing record is judged better by LLM."))

        print("=" * 40)
        print()

        return keep_new, None

    def mh_refine_record(self, prior: dict, draft: dict) -> Tuple[dict, bool]:
        """
        Apply Metropolis-Hastings local moves to refine the draft record.
        This implements q_edit(record'|record) + acceptance step to sharpen the draft, adding specificity and diversity to the corpus.
        """
        model = self.model
        profile = prior['profile']
        record_type = prior['record_type']
        situation = prior['social_context']
        form = prior['format'].replace('**', '')
        record = draft['text']
        accepted_records = [record]
        rejected_records = []

        mh_iterations = 2  # Number of MH refinement steps
        max_retries = 3
        for _ in range(mh_iterations):
            prompt = f"""Revise the following record to make it more specific and realistic.

**Record**:
{record}

Make sure to generate the '{record_type}' with less markdown formatting. Only output the revised record."""

            # Try to generate new record with retries
            for attempt in range(max_retries):
                try:
                    new_record = model.interact(prompt=prompt, temperature=0.7, max_tokens=8000)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"MH refinement attempt {attempt + 1} failed: {e}. Retrying...")
                        time.sleep(1.0 * (2 ** attempt))  # Exponential backoff
                    else:
                        logger.error(f"MH refinement failed after {max_retries} attempts: {e}")
                        raise RecordGenerationError(f"Failed to refine record: {e}") from e

            # Compare records and determine if we should keep the new one
            should_keep_new, new_embedding = self.compare_records(record, new_record)
        
            if should_keep_new:
                accepted_records.append(new_record)
                record = new_record
                # Add the new embedding to the history, maintaining max size (thread-safe)
                if new_embedding is not None:
                    with self._embeddings_lock:
                        self.previous_embeddings.append(new_embedding)
                        if len(self.previous_embeddings) > self.max_embeddings:
                            self.previous_embeddings = self.previous_embeddings[-self.max_embeddings:]
            else:
                rejected_records.append(new_record)

        record_id = get_text_hash(record)
        output = {'text': record, 'prompt': prompt, 'type': record_type, 'id': record_id, 'accepted_records': accepted_records, 'rejected_records': rejected_records}
        return output, False 

    def finalize_record(self, record: dict, form: str, attrs: dict, situation: dict) -> dict:
        record_type = record['type']
        situation_text = situation['text']
        prompt = f"Generate a specific and realistic '{record_type}' in English according to the information below.\n\n**Basic information**:\n{json.dumps(attrs, indent=4)}\n\n**Situation**: {situation_text}\n\n**Structure and tone of the '{record_type}'**: {form}\n\nMake sure to generate the '{record_type}' with less markdown formatting. Only output the generated '{record_type}'."
        model = self.model
        final_record = model.interact(prompt=prompt, temperature=1, max_tokens=7000)
        record_id = get_text_hash(final_record)
        output = {'text': final_record, 'prompt': prompt, 'type': record_type, 'id': record_id}
        output['draft_records'] = record['draft_records']
        output['draft_records'].append(record['text'])
        return output

    def flatten_dict(self, d, parent_key='', sep='.'):
        """
        Recursively flattens a nested dictionary.
        
        Args:
            d (dict): The dictionary to flatten.
            parent_key (str): The base key string for recursion (used internally).
            sep (str): The separator between keys.

        Returns:
            dict: A flattened dictionary.
        """
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self.flatten_dict(v, new_key, sep=sep))
            else:
                items[new_key] = v
        return items

    def generate_hierarchical_prior(self, seed: dict) -> Tuple[dict, bool]:
        """
        Generate the hierarchical prior z=(profile, record-type, social-context, format) following:
        p(z) = p(profile)p(record-type|profile)p(social-context|profile,record-type)p(format|record-type,social-context)
        
        Args:
            seed (dict): Initial seed values for profile generation
            
        Returns:
            Tuple[dict, bool]: 
                - Dictionary containing profile, record_type, social_context, and format
                - Error flag indicating if generation failed
        """
        # Step 1: Generate profile (p(profile))
        profile, error_flag = self.generate_profile_w_event(seed)
        if error_flag:
            return {"error": "[profile-gen] Failed to generate profile"}, True
            
        # Step 2: Select attribute to share and generate record type (p(record-type|profile))
        try:
            attr_to_share = self.select_attributes(profile['event_list'][0])
        except (ValueError, KeyError, IndexError) as e:
            logger.error(f"Error selecting attributes for profile {profile['id']}: {e}")
            print(Panel(f"{e}\nError in selecting attributes for profile {profile['id']}\n{profile['generated_profile_string']}", box=box.SIMPLE_HEAD, title="Error", expand=False, style="red"))
            return {"error": f"[attr-select] No valid attributes found during attribute selection: {e}", "profile_id": profile.get('id')}, True
        profile['attr_to_share'] = attr_to_share
        
        record_type = self.sample_record_types(attr_to_share, profile)
        if record_type is None:
            return {'error': "[record-type] Failed to sample record type", 'profile': profile, 'record_type': None, 'social_context': None, 'format': None}, True
        
        # Step 3: Generate social context (p(social-context|profile,record-type))
        situation = self.generate_situation(attr_to_share, profile, record_type)
        if "error_message" in situation:
            return {'error': situation['error_message'], 'profile': profile, 'record_type': record_type, 'social_context': None, 'format': None}, True
        
        # Step 4: Generate format (p(format|record-type,social-context))
        record_format = self.sample_format(record_type, situation['text'])
        if isinstance(record_format, dict) and "error_message" in record_format:
            return {'error': record_format['error_message'], 'profile': profile, 'record_type': record_type, 'social_context': situation, 'format': None}, True
        
        return {
            'profile': profile,
            'record_type': record_type,
            'social_context': situation,
            'format': record_format
        }, False

    def generate_privasis_data(self, seed) -> Tuple[List[dict], bool]:
        """
        Generate a single data point for Privasis dataset.
        1. Generate hierarchical prior z=(profile, record-type, social-context, format)
        2. Draft the record using the hierarchical prior (q_0)
        3. Run the MH algorithm to refine the draft (q_edit + acceptance)
        """
        error_flag = False
        
        # Generate hierarchical prior
        prior, error_flag = self.generate_hierarchical_prior(seed)
        if error_flag:
            return prior, error_flag
            
        # Try generating the record with one retry for validation failures
        max_record_attempts = 2
        validation_error = None
        
        for attempt in range(max_record_attempts):
            # Draft the record using the hierarchical prior
            draft, error_flag = self.draft_record(prior)
            if error_flag:
                return draft, error_flag

            # Run the MH algorithm
            record, error_flag = self.mh_refine_record(prior, draft)
            if error_flag:
                return record, error_flag

            # Validate the generated record
            record_text = record['text'].strip()
            validation_error = None
            
            # Check for LLM refusal
            if record_text.lower().startswith("i'm sorry") or record_text.lower().startswith("i am sorry"):
                validation_error = '[record-validate] LLM refusal'
                if attempt < max_record_attempts - 1:
                    logger.warning(f"LLM refusal detected for record {record['id']}, retrying (attempt {attempt + 1}/{max_record_attempts})")
                    continue
                else:
                    logger.warning(f"LLM refusal detected for record {record['id']} after {max_record_attempts} attempts")
                    error_record = {**record, 'error': validation_error}
                    return error_record, True
            
            # Check for minimum word count
            word_count = len(record_text.split())
            if word_count < 32:
                validation_error = f'[record-validate] Less than 32 words ({word_count} words)'
                if attempt < max_record_attempts - 1:
                    logger.warning(f"Record too short ({word_count} words) for record {record['id']}, retrying (attempt {attempt + 1}/{max_record_attempts})")
                    continue
                else:
                    logger.warning(f"Record too short ({word_count} words) for record {record['id']} after {max_record_attempts} attempts")
                    error_record = {**record, 'error': validation_error}
                    return error_record, True
            
            # Validation passed, break out of retry loop
            break

        # get the included attributes
        try:
            attributes = self.expand_attributes(prior['profile'], prior['social_context'], record)
        except (json.JSONDecodeError, KeyError, TypeError, ModelInteractionError) as e:
            logger.error(f"Error annotating attributes for record {record['id']}: {e}")
            print(Panel(f"{e}\nError in annotating attributes for record {record['id']}\n{record['text'][:500]}...", box=box.SIMPLE_HEAD, title="Error", expand=False, style="red"))
            error_record = {**record, 'error': f"[attr-expand] Annotation failed: {e}"}
            return error_record, True

        if "error_message" in attributes:
            logger.warning(f"Attribute expansion failed for record {record.get('id', 'unknown')}: {attributes['error_message']}")
            error_record = {**record, 'error': attributes['error_message']}
            return error_record, True

        record['attributes'] = attributes['attributes']
        record['grouped_attributes'] = attributes['grouped_attributes']

        # Finalize the record
        # commented out because diversity is not good enough
        # final_record = self.finalize_record(record, prior['format'], record['attributes'], prior['social_context'], model_name=model_name)
        # final_record['attributes'] = record['attributes']
        # final_record['grouped_attributes'] = record['grouped_attributes']
        
        if self.args.print:
            print(Panel(json.dumps(record['attributes']['event'], indent=4), box=box.SIMPLE_HEAD, title="Annotated attributes", expand=False))
            print(Panel(json.dumps(record['grouped_attributes'], indent=4), box=box.SIMPLE_HEAD, title="Grouped attributes", expand=False, style="blue"))
            print(Panel(prior['social_context']['text'], box=box.SIMPLE_HEAD, title="Situation", expand=False, style="green"))
            if not self.online:
                print(Panel(record['text'], box=box.SIMPLE_HEAD, title=prior['record_type'], expand=False, style="grey53"))

        texts = f"{prior['profile']['generated_profile_string']}\n\n{prior['social_context']['text']}\n\n{record['text']}"
        texts_id = get_text_hash(texts)

        final_output = {
            "profile": prior['profile'],
            "situation": prior['social_context'],
            "format": prior['format'],
            "record": record,
            "id": texts_id
        }

        return final_output, error_flag

    def _process_seed(self, idx: int, seed: dict) -> Tuple[int, dict, bool]:
        """Process a single seed and return the result.
        
        This method is designed to be called from multiple threads.
        
        Args:
            idx: Index of the seed
            seed: Seed data for generation
            
        Returns:
            Tuple of (index, data, error_flag)
        """
        data, error_flag = self.generate_privasis_data(seed)
        return idx, data, error_flag
    
    def _save_result(self, data: dict, error_flag: bool):
        """Save result to file with thread-safe locking.
        
        Args:
            data: Generated data to save
            error_flag: Whether generation encountered an error
        """
        with self._file_lock:
            if error_flag:
                # Ensure error data always has an 'error' key for debugging, and it comes first
                if isinstance(data, dict) and 'error' not in data and 'error_message' not in data:
                    logger.warning(f"Saving error without error message. Data keys: {list(data.keys())}")
                    data = {'error': '[unknown] No error message provided', **data}
                with open(self.error_file, 'a') as f:
                    f.write(json.dumps(data) + '\n')
            else:
                with open(self.output_file, 'a') as f:
                    f.write(json.dumps(data) + '\n')
        
        # Update statistics (thread-safe)
        with self._stats_lock:
            if error_flag:
                self._failure_count += 1
            else:
                self._success_count += 1

    def run(self):
        logs = read_logs(self.output_file)
        last_save_idx = logs.index[-1] if logs is not None else -1
        
        # Filter seeds that haven't been processed yet
        seeds_to_process = [(idx, seed) for idx, seed in enumerate(self.seeds) if idx > last_save_idx]
        
        if not seeds_to_process:
            print("All seeds already processed.")
            return
        
        # Use multithreading if num_workers > 1 and using vLLM server
        if self.num_workers > 1 and self.vllm_server_url:
            print(f"Running with {self.num_workers} parallel workers (vLLM server: {self.vllm_server_url})")
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(self._process_seed, idx, seed): idx 
                    for idx, seed in seeds_to_process
                }
                
                # Process results as they complete
                for future in tqdm(as_completed(futures), total=len(futures), desc="Generating records"):
                    original_idx = futures[future]
                    try:
                        idx, data, error_flag = future.result()
                        self._save_result(data, error_flag)
                    except Exception as e:
                        error_tb = traceback.format_exc()
                        logger.error(f"Error processing seed {original_idx}: {e}\n{error_tb}")
                        # Save error information with full context
                        self._save_result({
                            "error": f"[thread-exception] {type(e).__name__}: {e}",
                            "traceback": error_tb,
                            "seed_idx": original_idx
                        }, error_flag=True)
        else:
            # Single-threaded execution (original behavior)
            if self.num_workers > 1 and not self.vllm_server_url:
                print("Warning: --num-workers > 1 requires --vllm-server-url. Running single-threaded.")
            
            for idx, seed in tqdm(seeds_to_process, total=len(seeds_to_process), desc="Generating records"):
                data, error_flag = self.generate_privasis_data(seed)
                self._save_result(data, error_flag)

        # Print generation statistics
        total = self._success_count + self._failure_count
        if total > 0:
            success_rate = (self._success_count / total) * 100
            print("\n" + "=" * 50)
            print("Generation Statistics")
            print("=" * 50)
            print(f"  Succeeded: {self._success_count}")
            print(f"  Failed:    {self._failure_count}")
            print(f"  Total:     {total}")
            print(f"  Success Rate: {success_rate:.1f}%")
            print("=" * 50)


def main(args):
    generator = PrivasisGenerator(args)
    try:
        generator.run()
    finally:
        # Close the model client to release HTTP connection pools
        # This prevents the script from hanging after all workers are done
        if hasattr(generator, 'model') and hasattr(generator.model, 'close'):
            generator.model.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', type=str, default="v0", help="Run ID, which will also be used as the output file name under data/privasis/")
    parser.add_argument('--seeds_path', type=str, default=None, help="Path to seeds for profile generation")
    parser.add_argument('--profile_template_events', type=str, default="profile/template_profile_v0.8.txt", help="Path to profile generation template for events")
    parser.add_argument('--num_events', type=int, default=1, help="Number of events per profile")
    parser.add_argument('--n_seeds', type=int, default=10, help="Number of seeds to generate profiles for")
    parser.add_argument('--tolerance', type=int, default=3, help="Number of tries before accepting failure")
    parser.add_argument('--flags', type=str, default="", help="Flags to enable logic in testing")
    parser.add_argument('--event_dict_path', type=str, default='./data/events-attrs/consolidated_events.json', help="Path to JSON file containing events dictionary")
    parser.add_argument('--print', action='store_true', help="Print the generated outputs")
    parser.add_argument('--generator-model', type=str, required=True, help="The model name to use for generation (e.g., openai/gpt-oss-120b)")
    parser.add_argument('--embedding-model', type=str, default="Qwen/Qwen3-Embedding-0.6B", 
                       help="Embedding model to use. Supports Huggingface models (e.g., Qwen/Qwen3-Embedding-0.6B) or OpenAI models (e.g., text-embedding-3-small)")
    parser.add_argument('--embedding-device', type=str, default="cuda", 
                       help="Device for HF embedding model: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc. (default: cuda)")
    
    # vLLM-specific arguments
    parser.add_argument('--num_gpus', type=int, default=1, help="Number of GPUs for vLLM tensor parallelism")
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help="GPU memory utilization for vLLM (0.0-1.0)")
    parser.add_argument('--max_model_len', type=int, default=None, help="Maximum model context length for vLLM")
    
    # vLLM server arguments (for parallel processing)
    parser.add_argument('--vllm-server-url', type=str, default=None, 
                       help="URL of vLLM server (e.g., http://localhost:8000/v1). "
                            "If provided, connects to external vLLM server instead of loading model locally.")
    parser.add_argument('--num-workers', type=int, default=1,
                       help="Number of parallel worker threads for generation (default: 1). "
                            "Only effective when using --vllm-server-url.")
    
    args = parser.parse_args()
    main(args)
