#!/usr/bin/env python3
"""
Run a sanitization model on benchmark data and optionally evaluate results.
"""

import os
import json
import argparse
import logging
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import pandas as pd
from evaluate import SanitizationEvaluator, load_sanitized_results
import pickle
import time
import random

# Import model client
from model_client import UnifiedModelClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EVAL_OUTPUT_DIR = "benchmark_predictions"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class SanitizationModel(UnifiedModelClient):
    """Wrapper for sanitization models with custom post-processing.
    
    Extends UnifiedModelClient with sanitization-specific post-processing.
    """
    
    def __init__(self, *args, **kwargs):
        # Add custom postprocessing for sanitization
        kwargs['postprocess_fn'] = self._sanitization_postprocess
        super().__init__(*args, **kwargs)
    
    @staticmethod
    def _sanitization_postprocess(text: str) -> str:
        """Extract sanitized text from model output."""
        import re
        # Strip trailing prompt echo: model may append "**Sanitized Text:**" at the end
        text = re.sub(r'\s*\*{0,2}\s*Sanitized Text:\s*\*{0,2}\s*$', '', text).strip()
        # If "Sanitized Text:" still appears (as a prefix), extract content after it
        if "Sanitized Text:" in text:
            text = text.split("Sanitized Text:")[-1].strip()
        return text
    
def load_test_data(file_path: str = None, hf_subset: str = None, hf_split: str = None) -> List[Dict[str, Any]]:
    """
    Load test data from a JSONL file or nvidia/Privasis-Zero HuggingFace dataset.

    Args:
        file_path: Path to a local JSONL file
        hf_subset: Privasis-Zero dataset subset ('vanilla' or 'hard')
        hf_split: Privasis-Zero dataset split ('test' or 'validation')

    Returns:
        List of sanitized results
    """
    if hf_subset:
        from datasets import load_dataset
        ds = load_dataset("nvidia/Privasis-Zero", name=hf_subset, split=hf_split)
        df = ds.to_pandas()
    else:
        df = pd.read_json(file_path, lines=True)
    if 'sanitization_info' in df.columns:
        df['sanitized_record'] = df.apply(lambda row: {'sanitization_info': row['sanitization_info'], 'sanitized_text': row['text'], 'original_text': row['original_text'], 'smoothed_instruction': row['smoothed_instruction'], 'base_instruction': row['base_instruction'], 'id': row['id'], 'record_type': row['record']['type']}, axis=1)
    elif 'original_record' in df.columns:
        # Benchmark format: flat columns with original_record/sanitized_record as strings
        def _ensure_dict(val):
            if isinstance(val, str):
                return json.loads(val)
            return val if isinstance(val, dict) else {}

        def build_sanitization_info(row):
            info = _ensure_dict(row.get('grouped_annotated_attributes', {}))
            info['attrs_to_keep'] = _ensure_dict(row.get('attributes_to_keep', {}))
            target_attrs = {}
            for san_type in ['attributes_to_drop', 'attributes_to_abstract']:
                san_data = _ensure_dict(row.get(san_type, {}))
                if san_data and san_data.get('selected'):
                    selected = san_data['selected']
                    sanitization = san_type.replace('attributes_to_', '')
                    if isinstance(selected, list):
                        # Grouped format: [{attrs: {name: value}, group_name: ...}, ...]
                        for group in selected:
                            for attr_name, attr_value in group.get('attrs', {}).items():
                                target_attrs[attr_name] = {'value': attr_value, 'sanitization': sanitization}
                    elif isinstance(selected, dict):
                        # Flat format: {attr_name: value, ...}
                        for attr_name, attr_value in selected.items():
                            target_attrs[attr_name] = {'value': attr_value, 'sanitization': sanitization}
            info['individual_target_attributes'] = target_attrs
            return info

        df['sanitized_record'] = df.apply(lambda row: {'sanitization_info': build_sanitization_info(row), 'sanitized_text': row['sanitized_record'], 'original_text': row['original_record'], 'smoothed_instruction': row['smoothed_instruction'], 'base_instruction': row['base_instruction'], 'id': row['id'], 'record_type': row['record_type']}, axis=1)
    else:
        df['sanitized_record'] = df.apply(lambda row: {'sanitization_info': row['sanitized_record']['sanitization_info'], 'sanitized_text': row['sanitized_record']['text'], 'original_text': row['record']['text'], 'smoothed_instruction': row['smoothed_instruction'], 'base_instruction': row['base_instruction'], 'id': row['id'], 'record_type': row['record']['type']}, axis=1)
    return df['sanitized_record'].to_list()

def _output_dir() -> str:
    """Return the full path to the evaluation output directory."""
    return os.path.join(CURRENT_DIR, "outputs", EVAL_OUTPUT_DIR)

def get_checkpoint_path(output_filename: str) -> str:
    """Generate an absolute checkpoint path from an output filename."""
    if output_filename.endswith('.jsonl'):
        base = output_filename.replace('.jsonl', '_checkpoint.pkl')
    else:
        base = output_filename + "_checkpoint.pkl"
    return os.path.join(_output_dir(), base)

def load_checkpoint(checkpoint_path: str) -> List[Dict]:
    """Load processed data from checkpoint file (expects an absolute path)."""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint from {checkpoint_path}: {e}")
    return []

def save_checkpoint(processed_data: List[Dict], checkpoint_path: str):
    """Save processed data to checkpoint file (expects an absolute path)."""
    try:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(processed_data, f)
        logger.info(f"Checkpoint saved to: {checkpoint_path} ({len(processed_data)} records)")
    except Exception as e:
        logger.error(f"Failed to save checkpoint to {checkpoint_path}: {e}")

def generate_sanitized_texts(model: SanitizationModel, test_data: List[Dict], 
                           checkpoint_path: str = None, save_interval: int = 10, batch_size: int = 1, 
                           use_smoothed_instruction: bool = False, max_tokens: int = 9024,
                           temperature: float = 0.0) -> List[Dict]:
    """Generate sanitized texts using the model for test data that doesn't have sanitized text."""
    processed_data = []
    
    # Load existing checkpoint if available
    if checkpoint_path:
        processed_data = load_checkpoint(checkpoint_path)
        logger.info(f"Loaded {len(processed_data)} processed records from checkpoint {checkpoint_path}")
    
    # Calculate starting index based on existing processed data
    start_idx = len(processed_data)
    total_records = len(test_data)
    remaining_records = total_records - start_idx
    
    if remaining_records <= 0:
        logger.info("All records have been processed. No new processing needed.")
        return processed_data
    
    logger.info(f"Starting processing from index {start_idx} out of {total_records} total records")
    logger.info(f"Remaining records to process: {remaining_records}")
    
    start_time = time.time()
    batches_processed = 0

    for i in tqdm(range(start_idx, len(test_data), batch_size), desc="Processing batches"):
        batch_data = test_data[i:i + batch_size]
        sanitization_prompts = []
        original_texts = []

        for item in batch_data:
            original_text = item['original_text']
            original_texts.append(original_text)
            if use_smoothed_instruction:
                instruction = item['smoothed_instruction']
            else:
                instruction = item['base_instruction']

            # build sanitization prompts
            sanitization_prompt = f"**Sanitization Instruction:**\n{instruction}\nDo not output any explanation or other comment than the sanitized text.\n\n**Text to sanitize:**\n{original_text}\n\n**Sanitized Text:**"
            sanitization_prompts.append(sanitization_prompt)

        if len(sanitization_prompts) > 0:
            try:
                # Use the generate method (works for all model types)
                sanitized_texts = model.generate(
                    sanitization_prompts,
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                # Print a random sample for monitoring
                o, s = random.choice(list(zip(original_texts, sanitized_texts)))
                print(f"Original Text:\n{o}")
                print("-"*100)
                print(f"\033[94mSanitized Text:\n{s}\033[0m")
                print("="*100)
                print()

                for j, sanitized_text in enumerate(sanitized_texts):
                    processed_data.append({
                        **batch_data[j],
                        'model_sanitized_text': sanitized_text,
                    })

                batches_processed += 1

                # Save checkpoint periodically (based on batches in this run, not total count)
                if checkpoint_path and (batches_processed % save_interval == 0):
                    save_checkpoint(processed_data, checkpoint_path)

                    # Estimate remaining time
                    elapsed_time = time.time() - start_time
                    processed_count = len(processed_data) - start_idx
                    if processed_count > 0:
                        avg_time_per_record = elapsed_time / processed_count
                        remaining_time = avg_time_per_record * (total_records - len(processed_data))
                        logger.info(f"Progress: {len(processed_data)}/{total_records} records processed. "
                                  f"Estimated time remaining: {remaining_time/60:.1f} minutes")

            except Exception as e:
                logger.error(f"Error processing batch at index {i} (records {i}-{i+len(batch_data)-1}): {e}")
                # Save checkpoint on error
                if checkpoint_path:
                    save_checkpoint(processed_data, checkpoint_path)
                raise
    
    # Save final checkpoint
    if checkpoint_path:
        save_checkpoint(processed_data, checkpoint_path)
    
    total_time = time.time() - start_time
    logger.info(f"Processing completed in {total_time/60:.1f} minutes")
    
    return processed_data

def evaluate_sanitization(model_outputs: List[Dict], output_file: str, target_model_name: str = None, evaluator_model_name: str = "gpt-4.1", evaluator_vllm_server_url: str = None) -> Dict:
    """Evaluate the sanitization model."""
    # Initialize evaluator
    evaluator = SanitizationEvaluator(model_name=evaluator_model_name, vllm_server_url=evaluator_vllm_server_url)
    
    # Run evaluation
    print(f"Starting evaluation with model: {evaluator_model_name}")
    evaluation_results = evaluator.evaluate_batch(model_outputs, output_file)
    
    # Generate report
    evaluator.generate_detailed_report(evaluation_results, output_file, target_model_name=target_model_name)

def cleanup_checkpoint(checkpoint_path: str):
    """Remove checkpoint file after successful completion (expects an absolute path)."""
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
            logger.info(f"Cleaned up checkpoint file: {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up checkpoint file {checkpoint_path}: {e}")

def save_results(results: List[Dict], output_filename: str):
    """Save evaluation results."""
    output_path_dir = _output_dir()
    os.makedirs(output_path_dir, exist_ok=True)
    output_path = os.path.join(output_path_dir, output_filename)
    df = pd.DataFrame(results)
    df.to_json(output_path, orient='records', lines=True)
    logger.info(f"Results saved to: {output_path}")

def print_metrics(metrics: Dict):
    """Print evaluation metrics."""
    print("\n=== Sanitization Evaluation Metrics ===")
    print(f"Total Examples: {metrics['total_examples']}")
    print(f"Total Attributes Tested: {metrics['total_attributes_tested']}")
    print(f"Total Failed Attributes: {metrics['total_failed_attrs']}")
    print(f"Total Success Attributes: {metrics['total_success_attrs']}")
    print(f"Overall Success Rate: {metrics['overall_success_rate']:.2f}%")
    print(f"Average Success Rate per Record: {metrics['average_success_rate']:.2f}%")
    print(f"Complete Success Rate: {metrics['total_complete_success']} / {metrics['total_records']} ({metrics['complete_success_ratio']:.2f}%)")
    
    if metrics['total_attributes_tested'] > 0:
        failure_rate = (metrics['total_failed_attrs'] / metrics['total_attributes_tested']) * 100
        print(f"Failure Rate: {failure_rate:.2f}%")
    
    print("=" * 50)

def main():
    parser = argparse.ArgumentParser(description="Test and evaluate the trained sanitization model")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model")
    
    # Data arguments
    parser.add_argument("--test_data_path", type=str,
                       help="Path to test data JSONL file")
    parser.add_argument("--hf_subset", type=str, choices=["vanilla", "hard"],
                       help="Privasis-Zero dataset subset ('vanilla' or 'hard')")
    parser.add_argument("--hf_split", type=str, default="test", choices=["test", "validation"],
                       help="Privasis-Zero dataset split (default: 'test')")
    parser.add_argument("--sanitized_output_filename", type=str,
                       help="Path to save sanitized results")
    parser.add_argument("--use_existing_sanitized_output", action="store_true",
                       help="Use existing sanitized output if it exists")
    parser.add_argument("--evaluation_output_filename", type=str, required=True,
                       help="Path to save evaluation results")
    
    # Checkpoint arguments
    parser.add_argument("--checkpoint_filename", type=str,
                       help="Path to save/load checkpoint file for resuming processing")
    parser.add_argument("--save_interval", type=int, default=3,
                       help="Number of batches to process before saving checkpoint")
    
    # Generation arguments
    parser.add_argument("--max_length", type=int, default=9024,
                       help="Maximum generation length (max tokens)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for inference")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for inference")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature (0.0 for deterministic)")
    
    # vLLM-specific arguments
    parser.add_argument("--no_vllm", action="store_true",
                       help="Disable vLLM and use HuggingFace Transformers instead (vLLM is used by default)")
    parser.add_argument("--num_gpus", type=int, default=1,
                       help="Number of GPUs to use for vLLM tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                       help="GPU memory utilization for vLLM (0.0-1.0)")
    parser.add_argument("--max_model_len", type=int, default=None,
                       help="Maximum model context length for vLLM")
    parser.add_argument("--vllm-server-url", type=str, default=None,
                       help="URL of vLLM server (e.g., http://localhost:8000/v1). "
                            "If provided, connects to external vLLM server instead of loading model locally.")
    
    # Evaluation arguments
    parser.add_argument("--num_examples", type=int, default=None,
                       help="Number of examples to evaluate (for quick testing)")
    parser.add_argument("--evaluator_model_name", type=str, default="hf/openai/gpt-oss-120b",
                       help="Name of the evaluator model")
    parser.add_argument("--evaluator-vllm-server-url", type=str, default=None,
                       help="URL of vLLM server for the evaluator model (e.g., http://localhost:8000/v1). "
                            "If provided, connects to external vLLM server instead of loading model locally.")
    parser.add_argument("--skip-evaluation", action="store_true",
                       help="Skip evaluation phase (only run sanitization)")

    parser.add_argument("--use_smoothed_instruction", action="store_true",
                       help="Use smoothed instruction")
    
    args = parser.parse_args()

    # Determine whether to use vLLM (enabled by default)
    use_vllm = not args.no_vllm

    # Validate that a data source is provided when generating sanitized output
    if not args.use_existing_sanitized_output:
        if not args.test_data_path and not args.hf_subset:
            parser.error("Either --test_data_path or --hf_subset is required when not using --use_existing_sanitized_output")

    if args.use_existing_sanitized_output:
        logger.info(f"Loading sanitized output from {args.sanitized_output_filename}")
        sanitized_output_path = os.path.join(_output_dir(), args.sanitized_output_filename)
        result_data = load_sanitized_results(sanitized_output_path)
        if args.num_examples:
            result_data = result_data[:args.num_examples]
        logger.info(f"Loaded {len(result_data)} sanitized records")
    else:
        # Load model
        vllm_server_url = args.vllm_server_url
        logger.info(f"Loading model from {args.model_path}")
        if vllm_server_url:
            logger.info(f"Using vLLM server at: {vllm_server_url}")
        else:
            logger.info(f"Using vLLM: {use_vllm}, Num GPUs: {args.num_gpus}")
        model = SanitizationModel(
            model_path=args.model_path, 
            device=args.device,
            use_vllm=use_vllm,
            num_gpus=args.num_gpus,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            vllm_server_url=vllm_server_url,
        )
        
        try:
            # Load test data
            if args.hf_subset:
                logger.info(f"Loading test data from HuggingFace: nvidia/Privasis-Zero ({args.hf_subset}/{args.hf_split})")
            else:
                logger.info(f"Loading test data from {args.test_data_path}")
            test_data = load_test_data(
                file_path=args.test_data_path,
                hf_subset=args.hf_subset,
                hf_split=args.hf_split,
            )
            
            if args.num_examples:
                test_data = test_data[:args.num_examples]
                logger.info(f"Using {len(test_data)} examples for evaluation")
            
            # Generate sanitized texts if needed
            if args.checkpoint_filename:
                checkpoint_path = os.path.join(_output_dir(), args.checkpoint_filename)
            elif args.sanitized_output_filename:
                checkpoint_path = get_checkpoint_path(args.sanitized_output_filename)
            else:
                checkpoint_path = None
            if checkpoint_path:
                logger.info(f"Checkpoint path: {checkpoint_path}")
            
            result_data = generate_sanitized_texts(
                model, 
                test_data, 
                checkpoint_path=checkpoint_path,
                save_interval=args.save_interval,
                batch_size=args.batch_size,
                use_smoothed_instruction=args.use_smoothed_instruction,
                max_tokens=args.max_length,
                temperature=args.temperature,
            )
            save_results(result_data, args.sanitized_output_filename)
            
            # # Clean up checkpoint after successful completion
            # if checkpoint_path:
            #     cleanup_checkpoint(checkpoint_path)
        finally:
            # Close the model client to release HTTP connection pools
            # This prevents the script from hanging after processing is done
            if hasattr(model, 'close'):
                model.close()

    # Evaluate model (unless --skip-evaluation is set)
    if args.skip_evaluation:
        logger.info("Skipping evaluation phase (--skip-evaluation flag set)")
    else:
        logger.info("Starting evaluation...")
        evaluate_sanitization(result_data, args.evaluation_output_filename, target_model_name=args.model_path, evaluator_model_name=args.evaluator_model_name, evaluator_vllm_server_url=args.evaluator_vllm_server_url)
        logger.info("Evaluation completed!")

if __name__ == "__main__":
    main() 