import json
import argparse
import logging
import os
from typing import List, Dict, Any, Optional
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from model_client import load_model

logger = logging.getLogger(__name__)

EVAL_OUTPUT_DIR = "outputs/benchmark_scores"

class SanitizationEvaluator:
    """
    Evaluator for testing the effectiveness of sanitization methods.
    Based on the get_under_sanitized_attrs() method from PrivasisSanitizer.
    """
    
    def __init__(self, model_name: str = 'gpt-4.1', vllm_server_url: str = None):
        """
        Initialize the evaluator.
        
        Args:
            model_name: The LLM model to use for evaluation
            vllm_server_url: URL of vLLM server (e.g., http://localhost:8000/v1)
                            If provided, connects to external vLLM server instead of loading model locally
        """
        self.model_name = model_name
        self.vllm_server_url = vllm_server_url
        self.llm = load_model(model_name, vllm_server_url=vllm_server_url)

    def get_model_inference(self, target_attributes: dict, text: str) -> str:
        prompts = []
        attrs = []
        if "</think>" in text:
            text = text.split("</think>")[1]
        question_header = "Given the following text, please answer the question below."
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
        try:
            results = self.llm.batch_interact(prompts, temperature=0, max_tokens=512)
        except Exception as e:
            logger.warning("batch_interact failed, retrying: %s", e)
            results = self.llm.batch_interact(prompts, temperature=0, max_tokens=512)

        results = [result.lower() for result in results]
        return results
        
    @staticmethod
    def _value_in_text(value, text: str, match_all: bool = False) -> bool:
        """Check if a value appears in text via case-insensitive string matching.

        Args:
            value: scalar, list, or dict value to search for.
            text: text to search in.
            match_all: if True, require *all* items match (for retention);
                       if False, require *any* item match (for effectiveness).
        """
        text_lower = text.lower()
        if isinstance(value, list):
            comparator = all if match_all else any
            return comparator(str(item).lower() in text_lower for item in value)
        elif isinstance(value, dict):
            comparator = all if match_all else any
            return comparator(str(item).lower() in text_lower for item in value.values())
        return str(value).lower() in text_lower

    def evaluate_sanitization_effectiveness(self, original_text: str, sanitized_text: str, target_attributes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the effectiveness of the sanitization.
        """
        attribute_values = [(attr, info['value']) for attr, info in target_attributes.items() if info['value'] is not None]
        success_attrs = []
        failed_attrs = []

        # Step 1: Check exact string matching in sanitized_text
        attrs_found_in_text_verbatim = []
        attrs_not_found_in_text = []

        for (attr, value) in attribute_values:
            if self._value_in_text(value, sanitized_text):
                attrs_found_in_text_verbatim.append((attr, value))
            else:
                attrs_not_found_in_text.append((attr, value))

        # Add attributes found in text as failures (direct text leak)
        for attr, value in attrs_found_in_text_verbatim:
            failed_attrs.append({
                'attr': attr, 
                'value': value, 
                'failed_reason': 'direct_text_leak'
            })
        
        # Step 2: Get model inference only for attributes not found in text
        inference_from_sanitized_text = []
        attrs_not_in_text_and_inference_response = []
        if attrs_not_found_in_text:
            # Create a subset of attrs_to_keep for inference
            attrs_for_inference = {attr: target_attributes[attr] for attr, _ in attrs_not_found_in_text}
            try:
                inference_from_sanitized_text = self.get_model_inference(attrs_for_inference, sanitized_text)
            except Exception as e:
                logger.warning("get_model_inference failed, retrying: %s", e)
                inference_from_sanitized_text = self.get_model_inference(attrs_for_inference, sanitized_text)

            # Step 2-1: Check if the attribute values are in the inference response with exact string matching
            for (attr, value), inference_response in zip(attrs_not_found_in_text, inference_from_sanitized_text):
                value_in_inference_response = self._value_in_text(value, inference_response)

                if value_in_inference_response:
                    failed_attrs.append({
                        'attr': attr, 
                        'value': value, 
                        'failed_reason': 'inference_leak',
                        'inference_response_from_sanitized_text': inference_response,
                    })
                else:
                    attrs_not_in_text_and_inference_response.append((attr, value, inference_response))

            # Step 3: Use LLM to check whether the attribute value is closer to the inference response from sanitized text than the original response
            if attrs_not_in_text_and_inference_response:
                prompts = []
                for attr, value, inference_response_from_sanitized_text in attrs_not_in_text_and_inference_response:
                    prompt = self.build_which_response_closer_prompt(inference_response_from_sanitized_text, target_attributes[attr]['inference_from_original_record'], value)
                    prompts.append(prompt)
                try:
                    raw_results = self.llm.batch_interact(prompts, temperature=0.0)
                except Exception as e:
                    logger.warning("batch_interact failed, retrying: %s", e)
                    raw_results = self.llm.batch_interact(prompts, temperature=0.0)
                results = [(result.strip().lower() == "response1") or ("1" in result.strip().lower()) or ("both" in result.strip().lower()) for result in raw_results]
                for (attr, value, inference_response_from_sanitized_text), result, raw_result in zip(attrs_not_in_text_and_inference_response, results, raw_results):
                    if result:
                        failed_attrs.append({
                            'attr': attr, 
                            'value': value, 
                            'failed_reason': 'proximity_leak',
                            'raw_result': raw_result,
                            'inference_response_from_sanitized_text': inference_response_from_sanitized_text,
                            'inference_response_from_original_record': target_attributes[attr]['inference_from_original_record']
                        })
                    else:
                        success_attrs.append({
                            'attr': attr, 
                            'value': value, 
                            'inference_response_from_sanitized_text': inference_response_from_sanitized_text,
                        })

        # summarize the results
        success_rate = (len(success_attrs) / len(attribute_values) * 100) if attribute_values else None
        failed_reasons = [failed_attr['failed_reason'] for failed_attr in failed_attrs]
        _failed_reasons = list(set(failed_reasons))
        _failed_reasons.sort()
        failed_reasons_summary = {reason: failed_reasons.count(reason) for reason in _failed_reasons}
        
        # Calculate ratios
        total_failures = len(failed_reasons)
        failed_reasons_with_ratios = {}
        for reason, count in failed_reasons_summary.items():
            ratio = (count / total_failures * 100) if total_failures > 0 else 0
            failed_reasons_with_ratios[reason] = {'count': count, 'ratio': ratio}

        return {
            'failed_attrs': failed_attrs,
            'success_attrs': success_attrs,
            'success_rate': success_rate,
            'complete_success': success_rate is not None and success_rate == 100.0,
            'failed_reasons_summary': failed_reasons_with_ratios
        }

    def evaluate_sanitization_effectiveness_simple(self, original_text: str, sanitized_text: str, target_attributes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the effectiveness of the sanitization using only direct text leak checking with exact string matching.
        This is a simplified version that skips model inference and proximity checks.
        """
        attribute_values = [(attr, info['value']) for attr, info in target_attributes.items() if info['value'] is not None]
        success_attrs = []
        failed_attrs = []

        # Check exact string matching in sanitized_text
        for (attr, value) in attribute_values:
            if self._value_in_text(value, sanitized_text):
                failed_attrs.append({
                    'attr': attr, 
                    'value': value, 
                    'failed_reason': 'direct_text_leak'
                })
            else:
                success_attrs.append({
                    'attr': attr, 
                    'value': value
                })

        # Summarize the results
        success_rate = (len(success_attrs) / len(attribute_values) * 100) if attribute_values else None
        failed_reasons = [failed_attr['failed_reason'] for failed_attr in failed_attrs]
        _failed_reasons = list(set(failed_reasons))
        _failed_reasons.sort()
        failed_reasons_summary = {reason: failed_reasons.count(reason) for reason in _failed_reasons}
        
        # Calculate ratios
        total_failures = len(failed_reasons)
        failed_reasons_with_ratios = {}
        for reason, count in failed_reasons_summary.items():
            ratio = (count / total_failures * 100) if total_failures > 0 else 0
            failed_reasons_with_ratios[reason] = {'count': count, 'ratio': ratio}

        return {
            'failed_attrs': failed_attrs,
            'success_attrs': success_attrs,
            'success_rate': success_rate,
            'complete_success': success_rate is not None and success_rate == 100.0,
            'failed_reasons_summary': failed_reasons_with_ratios
        }
            

    def evaluate_sanitization_retention(self, original_text: str, sanitized_text: str, attrs_to_keep: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the retention of the sanitization.
        """
        attribute_values = [(attr, info['value']) for attr, info in attrs_to_keep.items() if info['value'] is not None]
        success_attrs = []
        failed_attrs = []
        
        # Step 1: Check exact string matching in sanitized_text
        found_in_text_verbatim = []
        not_found_in_text = []
        
        for attr, value in attribute_values:
            if self._value_in_text(value, sanitized_text, match_all=True):
                found_in_text_verbatim.append((attr, value))
            else:
                not_found_in_text.append((attr, value))

        # Add attributes found in text as successes
        for attr, value in found_in_text_verbatim:
            success_attrs.append({
                'attr': attr,
                'value': value,
                'success_reason': 'value_found_in_text_verbatim'
            })

        # Step 2: Get model inference only for attributes not found in text
        results_sanitized = []
        if not_found_in_text:
            # Create a subset of attrs_to_keep for inference
            attrs_for_inference = {attr: attrs_to_keep[attr] for attr, _ in not_found_in_text}
            results_sanitized = self.get_model_inference(attrs_for_inference, sanitized_text)

        # Step 3: Use LLM to check if attribute values are included in sanitized_text or inference response
        prompts = []
        for i, (attr, value) in enumerate(not_found_in_text):
            if i < len(results_sanitized):
                inference_response = results_sanitized[i]
                prompt = self.build_value_inclusion_prompt(value, sanitized_text, inference_response)
                prompts.append(prompt)

        results = self.llm.batch_interact(prompts, temperature=0.0)
        results = [result.strip().upper() == "YES" for result in results]

        for i, (attr, value) in enumerate(not_found_in_text):
            if results[i]:
                success_attrs.append({
                    'attr': attr,
                    'value': value,
                    'sanitized': results_sanitized[i],
                    'success_reason': 'value_found_in_text_using_inference'
                })
            else:
                failed_attrs.append({
                    'attr': attr,
                    'value': value,
                    'failed_reason': 'value_not_found'
                })

        # Summarize the success
        success_reasons = [success_attr['success_reason'] for success_attr in success_attrs]
        _success_reasons = list(set(success_reasons))
        _success_reasons.sort()
        success_reasons_summary = {reason: success_reasons.count(reason) for reason in _success_reasons}

        # Calculate ratios
        total_successes = len(success_attrs)
        success_reasons_with_ratios = {}
        for reason, count in success_reasons_summary.items():
            ratio = (count / total_successes * 100) if total_successes > 0 else 0
            success_reasons_with_ratios[reason] = {'count': count, 'ratio': ratio}

        success_rate = (len(success_attrs) / len(attribute_values) * 100) if attribute_values else None

        return {
            'failed_attrs': failed_attrs,
            'success_attrs': success_attrs,
            'success_rate': success_rate,
            'complete_success': success_rate is not None and success_rate == 100.0,
            'success_reasons_summary': success_reasons_with_ratios
        }

    def evaluate_sanitization_retention_simple(self, original_text: str, sanitized_text: str, attrs_to_keep: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simple version of retention evaluation using only exact string matching.
        This version only checks if attribute values are found verbatim in the sanitized text.
        """
        attribute_values = [(attr, info['value']) for attr, info in attrs_to_keep.items() if info['value'] is not None]
        success_attrs = []
        failed_attrs = []

        # Check exact string matching in sanitized_text
        for attr, value in attribute_values:
            if self._value_in_text(value, sanitized_text, match_all=True):
                success_attrs.append({
                    'attr': attr, 
                    'value': value, 
                    'success_reason': 'value_found_in_text_verbatim'
                })
            else:
                failed_attrs.append({
                    'attr': attr, 
                    'value': value, 
                    'failed_reason': 'value_not_found_in_text'
                })

        # Summarize the success
        success_reasons = [success_attr['success_reason'] for success_attr in success_attrs]
        _success_reasons = list(set(success_reasons))
        _success_reasons.sort()
        success_reasons_summary = {reason: success_reasons.count(reason) for reason in _success_reasons}

        # Calculate ratios
        total_successes = len(success_attrs)
        success_reasons_with_ratios = {}
        for reason, count in success_reasons_summary.items():
            ratio = (count / total_successes * 100) if total_successes > 0 else 0
            success_reasons_with_ratios[reason] = {'count': count, 'ratio': ratio}

        success_rate = (len(success_attrs) / len(attribute_values) * 100) if attribute_values else None

        return {
            'failed_attrs': failed_attrs,
            'success_attrs': success_attrs,
            'success_rate': success_rate,
            'complete_success': success_rate is not None and success_rate == 100.0,
            'success_reasons_summary': success_reasons_with_ratios
        }

    def evaluate_batch(self, sanitized_results: List[Dict[str, Any]], output_file: str, save_interval: int = 10, target_model_name: str = None, faster: bool = False) -> Dict[str, Any]:
        """
        Evaluate a batch of sanitized results with periodic saving and resume capability.
        
        Args:
            sanitized_results: List of sanitization results
            output_file: Path to save the evaluation results
            save_interval: How often to save results (every N records)
            faster: Whether to use the faster evaluation method using only exact string matching (this is for model validation)
            
        Returns:
            Dictionary with batch evaluation results
        """
        # Create output directory if it doesn't exist
        output_path = Path(EVAL_OUTPUT_DIR) / Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check for existing checkpoint
        checkpoint_file = output_path.with_suffix('.checkpoint.json')
        resume_from = 0

        # Initialize fresh state
        batch_results = {}
        total_under_sanitized = 0
        total_sanitization_target_attributes = 0
        total_retention_target_attributes = 0
        total_sanitization_correct_attrs = 0
        total_retention_correct_attrs = 0
        average_sanitization_success_rates = []
        average_retention_success_rates = []
        total_complete_success = 0
        total_sanitization_complete_success = 0
        total_retention_complete_success = 0
        overall_failed_reasons_summary = {}
        overall_success_reasons_summary = {}

        # Also try legacy pickle checkpoint for backwards compatibility
        legacy_checkpoint = output_path.with_suffix('.checkpoint')
        actual_checkpoint = None
        if checkpoint_file.exists():
            actual_checkpoint = checkpoint_file
        elif legacy_checkpoint.exists():
            actual_checkpoint = legacy_checkpoint

        if actual_checkpoint is not None:
            print(f"Found existing checkpoint at {actual_checkpoint}")
            try:
                if str(actual_checkpoint).endswith('.json'):
                    with open(actual_checkpoint, 'r') as f:
                        checkpoint_data = json.load(f)
                else:
                    import pickle
                    with open(actual_checkpoint, 'rb') as f:
                        checkpoint_data = pickle.load(f)
                batch_results = checkpoint_data['batch_results']
                total_under_sanitized = checkpoint_data['total_under_sanitized']
                total_sanitization_target_attributes = checkpoint_data['total_sanitization_target_attributes']
                total_retention_target_attributes = checkpoint_data['total_retention_target_attributes']
                total_sanitization_correct_attrs = checkpoint_data['total_sanitization_correct_attrs']
                total_retention_correct_attrs = checkpoint_data['total_retention_correct_attrs']
                average_sanitization_success_rates = checkpoint_data['average_sanitization_success_rates']
                average_retention_success_rates = checkpoint_data['average_retention_success_rates']
                total_complete_success = checkpoint_data.get('total_complete_success', 0)
                total_sanitization_complete_success = checkpoint_data['total_sanitization_complete_success']
                total_retention_complete_success = checkpoint_data['total_retention_complete_success']
                overall_failed_reasons_summary = checkpoint_data.get('overall_failed_reasons_summary', {})
                overall_success_reasons_summary = checkpoint_data.get('overall_success_reasons_summary', {})
                resume_from = len(checkpoint_data['batch_results'])
                print(f"Resuming from record {resume_from}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Starting fresh evaluation")

        def save_evaluation_checkpoint():
            """Save current evaluation progress to checkpoint file"""
            checkpoint_data = {
                'batch_results': batch_results,
                'total_under_sanitized': total_under_sanitized,
                'total_sanitization_target_attributes': total_sanitization_target_attributes,
                'total_retention_target_attributes': total_retention_target_attributes,
                'total_sanitization_correct_attrs': total_sanitization_correct_attrs,
                'total_retention_correct_attrs': total_retention_correct_attrs,
                'average_sanitization_success_rates': average_sanitization_success_rates,
                'average_retention_success_rates': average_retention_success_rates,
                'total_complete_success': total_complete_success,
                'total_sanitization_complete_success': total_sanitization_complete_success,
                'total_retention_complete_success': total_retention_complete_success,
                'complete_sanitization_success_rate': (total_sanitization_complete_success / total_sanitization_target_attributes * 100) if total_sanitization_target_attributes > 0 else 0.0,
                'complete_retention_success_rate': (total_retention_complete_success / total_retention_target_attributes * 100) if total_retention_target_attributes > 0 else 0.0,
                'overall_failed_reasons_summary': overall_failed_reasons_summary,
                'overall_success_reasons_summary': overall_success_reasons_summary
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            print(f"Checkpoint saved at record {len(batch_results)}")

        # only get the sanitized_results that has more than 1 attribute to keep
        target_sanitized_results = [result for result in sanitized_results if len(result['sanitization_info']['attrs_to_keep']) > 0]

        skipped_records = 0

        try:
            for i, result in enumerate(tqdm(target_sanitized_results[resume_from:], desc="Evaluating sanitization")):
                original_text = result['original_text']
                # Require an explicit model-produced sanitization. We deliberately do
                # NOT fall back to result['text'] (the ground-truth/reference sanitized
                # record), which would silently score the gold labels instead of the
                # model. Run model inference first (e.g. run_benchmark.py) to populate
                # 'model_sanitized_text'.
                sanitized_text = result.get('model_sanitized_text')
                if sanitized_text is None:
                    logger.warning(
                        "Record %s has no 'model_sanitized_text'; skipping "
                        "(refusing to fall back to the ground-truth sanitized text). "
                        "Run model inference to produce model outputs before evaluating.",
                        result.get('record_id', result.get('id', resume_from + i)),
                    )
                    skipped_records += 1
                    continue
                if isinstance(sanitized_text, dict):
                    if 'response' in sanitized_text:
                        sanitized_text = sanitized_text['response']
                sanitization_target_attributes = result['sanitization_info']['individual_target_attributes']
                # Compute inference_from_original_record if missing (e.g. benchmark data)
                attrs_missing_inference = {k: v for k, v in sanitization_target_attributes.items()
                                           if 'inference_from_original_record' not in v}
                if attrs_missing_inference:
                    inferences = self.get_model_inference(attrs_missing_inference, original_text)
                    for (attr_key, _), inference in zip(attrs_missing_inference.items(), inferences):
                        sanitization_target_attributes[attr_key]['inference_from_original_record'] = inference
                if faster:
                    sanitization_eval_result = self.evaluate_sanitization_effectiveness_simple(original_text, sanitized_text, sanitization_target_attributes)
                else:
                    sanitization_eval_result = self.evaluate_sanitization_effectiveness(original_text, sanitized_text, sanitization_target_attributes)

                attributes_to_keep = result['sanitization_info']['attrs_to_keep']
                if faster:
                    retention_eval_result = self.evaluate_sanitization_retention_simple(original_text, sanitized_text, attributes_to_keep)
                else:
                    retention_eval_result = self.evaluate_sanitization_retention(original_text, sanitized_text, attributes_to_keep)

                eval_result = {}
                eval_result['sanitization_eval_result'] = sanitization_eval_result
                eval_result['retention_eval_result'] = retention_eval_result
                eval_result['complete_success'] = sanitization_eval_result['complete_success'] and retention_eval_result['complete_success']
                eval_result['record_index'] = resume_from + i
                # if 'record_type' in result:
                #     eval_result['record_type'] = result['record_type']
                # elif "type" in result:
                #     eval_result['record_type'] = result['type']
                # eval_result['profile'] = result['profile']
                # eval_result['situation'] = result['situation']
                # eval_result['structure_tone'] = result['structure_tone']
                # eval_result['record'] = result['record']
                # eval_result['record_id'] = result['id']
                if "record_id" in result:
                    batch_results[result['record_id']] = eval_result
                elif 'id' in result:
                    batch_results[result['id']] = eval_result
                
                total_under_sanitized += len(eval_result['sanitization_eval_result']['failed_attrs'])
                total_sanitization_correct_attrs += len(eval_result['sanitization_eval_result']['success_attrs'])
                if eval_result['sanitization_eval_result']['success_rate'] is not None:
                    average_sanitization_success_rates.append(eval_result['sanitization_eval_result']['success_rate'])

                total_retention_correct_attrs += len(eval_result['retention_eval_result']['success_attrs'])
                if eval_result['retention_eval_result']['success_rate'] is not None:
                    average_retention_success_rates.append(eval_result['retention_eval_result']['success_rate'])

                # Track complete success cases
                if eval_result['complete_success']:
                    total_complete_success += 1
                if eval_result['sanitization_eval_result']['complete_success']:
                    total_sanitization_complete_success += 1
                if eval_result['retention_eval_result']['complete_success']:
                    total_retention_complete_success += 1
                total_sanitization_target_attributes += len(sanitization_target_attributes)
                total_retention_target_attributes += len(attributes_to_keep)
                
                # Update overall failed reasons summary
                for reason, data in eval_result['sanitization_eval_result']['failed_reasons_summary'].items():
                    if reason not in overall_failed_reasons_summary:
                        overall_failed_reasons_summary[reason] = {'count': 0, 'ratio': 0}
                    overall_failed_reasons_summary[reason]['count'] += data['count']

                # Update overall success reasons summary
                for reason, data in eval_result['retention_eval_result']['success_reasons_summary'].items():
                    if reason not in overall_success_reasons_summary:
                        overall_success_reasons_summary[reason] = {'count': 0, 'ratio': 0}
                    overall_success_reasons_summary[reason]['count'] += data['count']
                
                # Save checkpoint periodically
                if (resume_from + i + 1) % save_interval == 0:
                    save_evaluation_checkpoint()

        except KeyboardInterrupt:
            print("\nEvaluation interrupted. Saving checkpoint...")
            save_evaluation_checkpoint()
            raise
        except Exception as e:
            print(f"\nError during evaluation: {e}")
            print("Saving checkpoint before exiting...")
            save_evaluation_checkpoint()
            raise
        
        # Save final checkpoint
        save_evaluation_checkpoint()

        if skipped_records > 0:
            logger.warning(
                "Skipped %d/%d records with no 'model_sanitized_text'. These were "
                "NOT scored against the ground truth. Run model inference first if "
                "you intended to evaluate a model's outputs.",
                skipped_records, len(target_sanitized_results),
            )

        # Remove checkpoint file after successful completion
        # if checkpoint_file.exists():
        #     checkpoint_file.unlink()
        #     print("Evaluation completed successfully. Checkpoint file removed.")
        
        overall_sanitization_success_rate = (total_sanitization_correct_attrs / total_sanitization_target_attributes * 100) if total_sanitization_target_attributes > 0 else 0.0
        overall_average_sanitization_success_rate = sum(average_sanitization_success_rates) / len(average_sanitization_success_rates) if average_sanitization_success_rates else 0.0
        complete_sanitization_success_rate = (total_sanitization_complete_success / len(target_sanitized_results) * 100) if len(target_sanitized_results) > 0 else 0.0

        overall_retention_success_rate = (total_retention_correct_attrs / total_retention_target_attributes * 100) if total_retention_target_attributes > 0 else 0.0
        overall_average_retention_success_rate = sum(average_retention_success_rates) / len(average_retention_success_rates) if average_retention_success_rates else 0.0
        complete_retention_success_rate = (total_retention_complete_success / len(target_sanitized_results) * 100) if len(target_sanitized_results) > 0 else 0.0
        
        # Calculate overall ratios for failed reasons
        total_overall_failures = sum(data['count'] for data in overall_failed_reasons_summary.values())
        for reason in overall_failed_reasons_summary:
            overall_failed_reasons_summary[reason]['ratio'] = (overall_failed_reasons_summary[reason]['count'] / total_overall_failures * 100) if total_overall_failures > 0 else 0

        # Calculate overall ratios for success reasons
        total_overall_successes = sum(data['count'] for data in overall_success_reasons_summary.values())
        for reason in overall_success_reasons_summary:
            overall_success_reasons_summary[reason]['ratio'] = (overall_success_reasons_summary[reason]['count'] / total_overall_successes * 100) if total_overall_successes > 0 else 0
        
        return {
            'batch_results': batch_results,
            'summary': {
                'total_records': len(target_sanitized_results),
                'total_under_sanitized_attributes': total_under_sanitized,
                'total_sanitization_target_attributes': total_sanitization_target_attributes,
                'total_retention_target_attributes': total_retention_target_attributes,
                'total_sanitization_correct_attrs': total_sanitization_correct_attrs,
                'total_retention_correct_attrs': total_retention_correct_attrs,

                'overall_sanitization_success_rate': overall_sanitization_success_rate,
                'average_sanitization_success_rate': overall_average_sanitization_success_rate,

                'overall_retention_success_rate': overall_retention_success_rate,
                'average_retention_success_rate': overall_average_retention_success_rate,

                'complete_sanitization_success_rate': complete_sanitization_success_rate,
                'complete_retention_success_rate': complete_retention_success_rate,
                'total_complete_success': total_complete_success,
                'complete_success_rate': (total_complete_success / len(target_sanitized_results) * 100) if len(target_sanitized_results) > 0 else 0.0,
                'total_sanitization_complete_success': total_sanitization_complete_success,
                'total_retention_complete_success': total_retention_complete_success,
                'overall_failed_reasons_summary': overall_failed_reasons_summary,
                'overall_success_reasons_summary': overall_success_reasons_summary
            }
        }
    
    def generate_detailed_report(self, evaluation_results: Dict[str, Any], output_file: str, target_model_name: str = None):
        """
        Generate a detailed evaluation report.
        
        Args:
            evaluation_results: Results from evaluate_batch
            output_file: Path to save the report
        """
        output_path = Path(EVAL_OUTPUT_DIR) / Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'evaluation_metadata': {
                'evaluator_model': self.model_name,
                'evaluation_timestamp': pd.Timestamp.now().isoformat()
            },
            'summary': evaluation_results['summary'],
            'batch_results': evaluation_results['batch_results']
        }
        
        # Save detailed report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        if target_model_name:
            print(f"Target Model: {target_model_name}")
        print(f"Run ID: {output_file}")
        print(f"Evaluator model: {self.model_name}")
        print(f"Total records evaluated: {evaluation_results['summary']['total_records']}")
        print(f"Total sanitization target attributes: {evaluation_results['summary']['total_sanitization_target_attributes']}")
        print(f"Total retention target attributes: {evaluation_results['summary']['total_retention_target_attributes']}")
        print(f"Total sanitization correct attributes: {evaluation_results['summary']['total_sanitization_correct_attrs']}")
        print(f"Total retention correct attributes: {evaluation_results['summary']['total_retention_correct_attrs']}")
        print(f"Total under-sanitized attributes found: {evaluation_results['summary']['total_under_sanitized_attributes']}")
        
        print("-"*50)
        print(f"Overall sanitization success rate: {evaluation_results['summary']['overall_sanitization_success_rate']:.2f}%")
        print(f"Average sanitization success rate per record: {evaluation_results['summary']['average_sanitization_success_rate']:.2f}%")
        print(f"Complete sanitization success rate: {evaluation_results['summary']['total_sanitization_complete_success']} / {evaluation_results['summary']['total_records']} ({evaluation_results['summary']['complete_sanitization_success_rate']:.2f}%)")
        print("-"*50)
        print(f"Overall retention success rate: {evaluation_results['summary']['overall_retention_success_rate']:.2f}%")
        print(f"Average retention success rate per record: {evaluation_results['summary']['average_retention_success_rate']:.2f}%")
        print(f"Complete retention success rate: {evaluation_results['summary']['total_retention_complete_success']} / {evaluation_results['summary']['total_records']} ({evaluation_results['summary']['complete_retention_success_rate']:.2f}%)")
        print("-"*50)
        print(f"Total complete success: {evaluation_results['summary']['total_complete_success']} / {evaluation_results['summary']['total_records']} ({float(evaluation_results['summary']['total_complete_success']) / float(evaluation_results['summary']['total_records']) * 100:.2f}%)")

        # Print failed reasons summary
        if 'overall_failed_reasons_summary' in evaluation_results['summary'] and evaluation_results['summary']['overall_failed_reasons_summary']:
            print(f"Failed Reasons Summary:")
            # Sort alphabetically by reason name
            sorted_reasons = sorted(evaluation_results['summary']['overall_failed_reasons_summary'].items(), 
                                  key=lambda x: x[0])
            for reason, data in sorted_reasons:
                print(f"  {reason}: {data['count']} ({data['ratio']:.1f}%)")
        
        print("="*50)
        
        # # Show examples of under-sanitized attributes
        # under_sanitized_examples = []
        # for record_id, result in evaluation_results['batch_results'].items():
        #     if 'error' not in result and result['sanitization_eval_result']['failed_attrs']:
        #         under_sanitized_examples.extend(result['sanitization_eval_result']['failed_attrs'])
        
        # if under_sanitized_examples:
        #     print(f"\nExamples of under-sanitized attributes found:")
        #     for i, example in enumerate(under_sanitized_examples[:5]):  # Show first 5 examples
        #         attr_name = example['attr']
        #         attr_info = example['value']
        #         if 'inference_response' in example:
        #             print(f"  {i+1}. {attr_name}: {attr_info}\n- [Failed reason]: {example['failed_reason'].replace('_', ' ')}\n- [Model inference from the sanitized text]: {example['inference_response']}")
        #         else:
        #             print(f"  {i+1}. {attr_name}: {attr_info}\n- [Failed reason]: {example['failed_reason'].replace('_', ' ')}")
        #         print("-"*50)

        # # show where retention failed
        # for record_id, result in evaluation_results['batch_results'].items():
        #     if 'error' not in result and not result['retention_eval_result']['complete_success']:
        #         print(f"Sanitization retention has failed for record {record_id}")
        #         print(f"  {result['retention_eval_result']['failed_attrs']}")
        
        print(f"\nDetailed report saved to: {output_path}")

    def is_same_response(self, response1: str, response2: str) -> bool:
        """
        Use LLM to determine if two responses are basically the same.

        Args:
            response1: First response to compare
            response2: Second response to compare

        Returns:
            True if responses are essentially the same, False otherwise
        """
        prompt = f"""Given two responses to the same question, determine if they are essentially the same answer.

Response 1: "{response1}"
Response 2: "{response2}"

Consider the following when comparing:
- Do they convey the same core information?
- Do they reveal the same level of detail?
- Ignore minor differences in wording, capitalization, or formatting

Answer with only "YES" if they are essentially the same response, or "NO" if they are meaningfully different. Do not add any other comments.

Answer:"""

        result = self.llm.interact(prompt, temperature=0.0)
        return result.strip().upper() == "YES"

    def which_response_closer(self, response1: str, response2: str, attribute_value: str) -> str:
        """
        Use LLM to determine which response is closer to the actual attribute value.

        Args:
            response1: First response to compare
            response2: Second response to compare
            attribute_value: The actual/correct attribute value

        Returns:
            "response1", "response2", or "neither" indicating which response is closer to the attribute value
        """
        prompt = f"""Given two responses to the same question and the actual correct answer, determine which response is closer to the correct answer.

Response 1: "{response1}"
Response 2: "{response2}"
Actual correct answer: "{attribute_value}"

Consider the following when comparing:
- Which response contains more accurate information?
- Which response is more specific and matches the actual value?
- Which response reveals more of the correct details?
- If neither response is close to the actual value, choose "neither"

Answer with only "response1", "response2", or "neither". Do not add any other comments.

Answer:"""

        result = self.llm.interact(prompt, temperature=0.0)
        result = result.strip().lower()
        if result in ["response1", "response2", "neither"]:
            return result
        else:
            logger.warning("Unexpected response from LLM: %s", result)
            return "neither"

    def build_which_response_closer_prompt(self, response1: str, response2: str, attribute_value: str) -> str:
        """
        Use LLM to determine which response is closer to the actual attribute value.
        
        Args:
            response1: First response to compare
            response2: Second response to compare
            attribute_value: The actual/correct attribute value
            model_name: The LLM model to use for comparison
            
        Returns:
            "response1", "response2", "both", or "neither" indicating which response is closer to the attribute value
        """
        prompt = f"""Given two responses to the same question and the actual correct answer, determine which response is closer to the correct answer.

Response 1: "{response1}"
Response 2: "{response2}"
Actual correct answer: "{attribute_value}"

Consider the following when comparing:
- Which response contains more accurate information?
- Which response is more specific and matches the actual value?
- Which response reveals more of the correct details?
- If both responses are close to the actual value, choose "both"
- If neither response is close to the actual value, choose "neither"

Answer with only "response1", "response2", "both", or "neither". Do not add any other comments.

Answer:"""
        return prompt

    def build_value_inclusion_prompt(self, value: Any, sanitized_text: str, inference_response: str) -> str:
        """
        Use LLM to check if the attribute value is included in sanitized_text or inference response.
        
        Args:
            value: The attribute value to check
            sanitized_text: The sanitized text
            inference_response: The model's inference response
            
        Returns:
            True if the value is included, False otherwise
        """
        # Convert value to string for the prompt
        if isinstance(value, list):
            value_str = ", ".join(str(item) for item in value)
        elif isinstance(value, dict):
            value_str = ", ".join(str(item) for item in value.values())
        else:
            value_str = str(value)
        
        prompt = f"""Given the following sanitized text and inference response, determine if the specific information is included in either of them.

Information to check:
"{value_str}"

Sanitized text:
{sanitized_text}

Inference response:
{inference_response}

Consider the following:
- Check if the specific information is present in either text
- If the specific information is present in either text, answer "YES"
- If the specific information is not present in either text, answer "NO"

Answer with only "YES" or "NO". Do not add any other comments.

Answer:"""

        return prompt


def load_sanitized_results(file_path: str = None, n_records: int = None,
                           hf_subset: str = None, hf_split: str = None) -> List[Dict[str, Any]]:
    """
    Load sanitized results from a JSONL file or HuggingFace dataset.

    Args:
        file_path: Path to a local JSONL file
        hf_subset: HuggingFace dataset subset ('vanilla' or 'hard')
        hf_split: HuggingFace dataset split ('test' or 'validation')
        n_records: Maximum number of records to load

    Returns:
        List of sanitized results
    """
    if hf_subset:
        from datasets import load_dataset
        ds = load_dataset("nvidia/Privasis-Zero", name=hf_subset, split=hf_split)
        df = ds.to_pandas()
    else:
        df = pd.read_json(file_path, lines=True)

    if 'original_record' in df.columns:
        # Benchmark format (Privasis-Zero): flat columns with original_record/sanitized_record as strings
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
                        for group in selected:
                            for attr_name, attr_value in group.get('attrs', {}).items():
                                target_attrs[attr_name] = {'value': attr_value, 'sanitization': sanitization}
                    elif isinstance(selected, dict):
                        for attr_name, attr_value in selected.items():
                            target_attrs[attr_name] = {'value': attr_value, 'sanitization': sanitization}
            info['individual_target_attributes'] = target_attrs
            return info

        # Raw benchmark contains only the ground-truth 'sanitized_record'; there is no
        # model output here. We store it as 'reference_sanitized_text' (NOT
        # 'model_sanitized_text') so the evaluator skips these records instead of
        # scoring the gold labels. Run model inference (run_benchmark.py) to evaluate.
        df['sanitized_record'] = df.apply(lambda row: {
            'sanitization_info': build_sanitization_info(row),
            'reference_sanitized_text': row['sanitized_record'],
            'original_text': row['original_record'],
            'smoothed_instruction': row['smoothed_instruction'],
            'base_instruction': row['base_instruction'],
            'id': row['id'],
            'record_type': row['record_type'],
        }, axis=1)
    elif "sanitized_record" in df.columns:
        # sanitize.py output format: sanitized_record['text'] is the model-produced
        # sanitization, which is what we evaluate.
        df['sanitized_record'] = df.apply(lambda row: {**row['sanitized_record'], 'model_sanitized_text': row['sanitized_record'].get('text'), 'original_text': row['record']['text'], 'base_instruction': row['base_instruction'], 'smoothed_instruction': row['smoothed_instruction'], 'id': row['id']}, axis=1)
    else:
        # Prediction format (run_benchmark.py output): carry model_sanitized_text
        # through explicitly. If it is absent, leave it absent so the evaluator skips
        # the record rather than silently scoring the ground-truth 'sanitized_text'.
        df['sanitized_record'] = df.apply(lambda row: {'model_sanitized_text': row.get('model_sanitized_text'),
                                                       'sanitization_info': row['sanitization_info'],
                                                       'base_instruction': row['base_instruction'],
                                                       'smoothed_instruction': row['smoothed_instruction'],
                                                       'original_text': row['original_text'], 'id': row['id']}, axis=1)
    if n_records is None:
        return df['sanitized_record'].to_list()
    else:
        return df['sanitized_record'].to_list()[:n_records]


def resume_evaluation(checkpoint_file: str) -> Dict[str, Any]:
    """
    Load evaluation state from a checkpoint file (JSON or legacy pickle).

    Args:
        checkpoint_file: Path to the checkpoint file

    Returns:
        Dictionary with checkpoint data
    """
    checkpoint_path = Path(checkpoint_file)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

    try:
        if str(checkpoint_path).endswith('.json'):
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
        else:
            import pickle
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
        print(f"Loaded checkpoint with {len(checkpoint_data['batch_results'])} completed evaluations")
        return checkpoint_data
    except Exception as e:
        raise Exception(f"Error loading checkpoint: {e}") from e


def main():
    parser = argparse.ArgumentParser(description='Evaluate sanitization effectiveness')
    parser.add_argument('--input-file', default=None, help='Path to JSONL file with sanitized results (if not using HuggingFace dataset)')
    parser.add_argument('--hf-subset', type=str, default=None, choices=['vanilla', 'hard'],
                        help="Privasis-Zero dataset subset: 'vanilla' or 'hard' (default: load from HuggingFace)")
    parser.add_argument('--hf-split', type=str, default='test', choices=['test', 'validation'],
                        help="Privasis-Zero dataset split: 'test' or 'validation' (default: test)")
    parser.add_argument('--output-file', required=True, help='Path to save evaluation report')
    parser.add_argument('--evaluator-model-name', default='nvdev/openai/gpt-oss-120b', help='LLM model to use for evaluation')
    parser.add_argument('--vllm-server-url', default=None, help='URL of vLLM server (e.g., http://localhost:8000/v1)')
    parser.add_argument('--save-interval', type=int, default=5, help='How often to save checkpoint (every N records)')
    parser.add_argument('--resume', action='store_true', help='Force resume from existing checkpoint')
    parser.add_argument('--n-records', type=int, default=None, help='Number of records to evaluate')
    parser.add_argument('--faster', action='store_true', help='Use faster evaluation with only exact string matching (no LLM inference)')

    args = parser.parse_args()

    if not args.input_file and not args.hf_subset:
        parser.error("Either --input-file or --hf-subset is required")

    # Check if we should force resume
    checkpoint_file = Path(EVAL_OUTPUT_DIR) / Path(args.output_file).with_suffix('.checkpoint.json')
    if args.resume and not checkpoint_file.exists():
        print(f"Warning: --resume flag specified but no checkpoint found at {checkpoint_file}")
        print("Starting fresh evaluation")
    elif args.resume and checkpoint_file.exists():
        print(f"Resuming from checkpoint: {checkpoint_file}")
        checkpoint_data = resume_evaluation(str(checkpoint_file))
        print(f"Checkpoint shows {len(checkpoint_data['batch_results'])} completed evaluations")

    # Load sanitized results
    if args.hf_subset:
        print(f"Loading sanitized results from HuggingFace: nvidia/Privasis-Zero ({args.hf_subset}/{args.hf_split})")
        sanitized_results = load_sanitized_results(hf_subset=args.hf_subset, hf_split=args.hf_split, n_records=args.n_records)
    else:
        print(f"Loading sanitized results from: {args.input_file}")
        sanitized_results = load_sanitized_results(file_path=args.input_file, n_records=args.n_records)
    print(f"Loaded {len(sanitized_results)} records")

    # Initialize evaluator
    evaluator = SanitizationEvaluator(model_name=args.evaluator_model_name, vllm_server_url=args.vllm_server_url)

    # Run evaluation
    print(f"Starting evaluation with evaluator model: {args.evaluator_model_name}")
    print(f"Checkpoint will be saved every {args.save_interval} records")
    evaluation_results = evaluator.evaluate_batch(sanitized_results, args.output_file, args.save_interval, faster=args.faster)

    # Generate report
    evaluator.generate_detailed_report(evaluation_results, args.output_file)


if __name__ == "__main__":
    main()