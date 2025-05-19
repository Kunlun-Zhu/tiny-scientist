import argparse
import json
import os
import traceback
from multiprocessing import Pool, Manager, Lock, cpu_count
from functools import partial
from datetime import datetime # For unique output sub-directories
from typing import List, Dict, Optional, Any
import logging
import sys

# TinyScientist imports
from tiny_scientist import Thinker, SafetyEvaluator
from tiny_scientist.prompts.prompts import AttackPrompts, DefensePrompts
from tiny_scientist.utils.llm import create_client

# Default model names (can be overridden by args)
DEFAULT_THINKER_MODEL = "gpt-3.5-turbo"
DEFAULT_EVALUATOR_MODEL = "gpt-4o" # SafetyEvaluator is used by Thinker

# --- Worker Function ---
def process_task_for_idea_generation(task_info_tuple, common_args, overall_output_dir_base, file_lock):
    """
    Processes a single task (intent + scenario configuration) to generate a research idea using Thinker.
    Writes the result to a shared JSONL file.
    """
    task_original_index, task_config = task_info_tuple # task_config is what load_tasks generates per scenario
    task_id = task_config.get("id", f"task_{task_original_index}_{task_config.get('scenario', 'unknown').lower()}")
    intent = task_config.get("intent")
    domain = task_config.get("domain")
    scenario = task_config.get("scenario")

    entry_to_write = {
        "task_id": task_id,
        "original_intent": intent,
        "domain": domain,
        "scenario": scenario,
        "thinker_model": common_args.thinker_model_name,
        "evaluator_model": common_args.evaluator_model_name, # For SafetyEvaluator inside Thinker
        "l2_defense_enabled_for_run": common_args.enable_defense_layer_2, # Global L2 flag for the run
        "status": "failed_setup", # Default status
        "thinker_output": None,
        "error_details": None,
        "traceback": None
    }

    if not intent or not domain:
        print(f"[WARNING] Worker {os.getpid()}: Task ID {task_id} is missing 'intent' or 'domain'. Skipping.")
        entry_to_write["error_details"] = "Missing intent or domain in task_config."
        entry_to_write["status"] = "skipped_missing_data"
    else:
        print(f"[INFO] Worker {os.getpid()}: Processing Task ID {task_id} (Scenario: {scenario}, Domain: {domain})")
        # Per-task artifact directory (optional, Thinker might not save files directly in this flow)
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") # Added microseconds for more uniqueness
        # current_task_artifact_dir = os.path.join(overall_output_dir_base, f"{task_id}_{timestamp}")
        # os.makedirs(current_task_artifact_dir, exist_ok=True)
        # entry_to_write["task_artifact_directory"] = current_task_artifact_dir

        try:
            # Determine the model for Thinker based on whether malicious agent is enabled for this specific task
            # MaliciousAgent, if created by Thinker, will inherit Thinker's model and client.
            thinker_model_for_this_scenario = common_args.thinker_model_name # Default (e.g., gpt-3.5-turbo)
            if task_config.get("malicious_agent_enabled", False):
                # For MALICIOUS_ATTACK scenarios, we want the MaliciousAgent (and thus Thinker that creates it)
                # to use the more capable model, which we've set as evaluator_model_name (e.g., gpt-4o).
                thinker_model_for_this_scenario = common_args.evaluator_model_name 
            
            thinker_llm_client, thinker_model_name_actual = create_client(model=thinker_model_for_this_scenario)
            
            # SafetyEvaluator always uses the designated evaluator_model_name
            evaluator_llm_client, evaluator_model_name_actual = create_client(model=common_args.evaluator_model_name)

            safety_evaluator = SafetyEvaluator(
                llm_client=evaluator_llm_client,
                model_name=evaluator_model_name_actual
            )

            # Configure Thinker based on scenario from task_config
            enable_malicious = task_config.get("malicious_agent_enabled", False)
            enable_defense_l1 = task_config.get("defense_agent_l1_enabled", False)
            enable_defense_l2_for_thinker = task_config.get("defense_agent_l2_enabled", False)

            thinker = Thinker(
                llm_client=thinker_llm_client, 
                model_name=thinker_model_name_actual, # This is now conditional (gpt-3.5 or gpt-4o)
                initial_research_intent=intent,
                domain=domain,
                safety_evaluator=safety_evaluator, # SafetyEvaluator always uses its specific model (e.g. gpt-4o)
                enable_malicious_agents=enable_malicious,
                enable_defense_agent=enable_defense_l1,
                enable_defense_agent_layer_2=enable_defense_l2_for_thinker,
                attack_prompt_template=task_config.get("attack_prompt_template"),
                defense_prompt_template=task_config.get("defense_prompt_template_l1"),
                defense_prompt_template_layer_2=task_config.get("defense_prompt_template_l2"),
                tools=[], iter_num=1,
            )

            thinker_output_dict = thinker.think()

            entry_to_write["thinker_output"] = thinker_output_dict
            if thinker_output_dict.get("error"):
                entry_to_write["error_details"] = f"Thinker reported an error: {thinker_output_dict['error']}"
                entry_to_write["status"] = "failed_thinker_reported_error"
                print(f"[ERROR] Worker {os.getpid()}: Task ID {task_id} - Thinker error: {thinker_output_dict['error']}")
            else:
                entry_to_write["status"] = "success"
                print(f"[SUCCESS] Worker {os.getpid()}: Task ID {task_id} completed.")

        except Exception as e:
            print(f"[ERROR] Worker {os.getpid()}: Unhandled exception processing Task ID {task_id}: {e}")
            entry_to_write["error_details"] = str(e)
            entry_to_write["traceback"] = traceback.format_exc()
            entry_to_write["status"] = "failed_processing_exception"

    # Write result to the shared JSONL output file
    if entry_to_write:
        try:
            file_lock.acquire()
            with open(common_args.output_file, 'a', encoding='utf-8') as outfile:
                outfile.write(json.dumps(entry_to_write) + "\n")
                outfile.flush()
        except Exception as e:
            print(f"[ERROR] Worker {os.getpid()}: Failed to write result for Task ID {entry_to_write.get('task_id')} to file: {e}")
        finally:
            file_lock.release()
            
    return entry_to_write.get("status", "unknown_status")


# --- Task Loading Function ---
def load_input_file(filepath: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Loads the input JSONL file, supporting an optional limit on the number of records."""
    records = []
    try:
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                try:
                    records.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    logging.warning(f"Skipping malformed JSON line in {filepath}: {line.strip()}")
    except FileNotFoundError:
        logging.error(f"Input file not found: {filepath}")
        sys.exit(1)
    return records

def generate_scenario_tasks(
    base_tasks: List[Dict[str, str]],
    attack_prompt_template_map: Dict[str, str],
    defense_prompt_template_map: Dict[str, str],
    defense_prompt_template_layer_2: Optional[str],
    enable_defense_layer_2_globally: bool,
    scenarios_to_run: List[str] # New parameter
) -> List[Dict[str, Any]]:
    """
    Generates a list of specific scenario tasks from the base tasks,
    now considering which scenarios to run.
    """
    scenario_tasks = []
    for base_task in base_tasks:
        intent = base_task.get("Prompt")
        domain = base_task.get("Domain")
        base_task_id = base_task.get("TaskID", intent[:50].replace(" ", "_")) # Use TaskID if present

        if not intent or not domain:
            logging.warning(
                f"[WARNING] Base task '{base_task_id}' in {base_task.get('filepath', 'input file')} is missing 'Prompt' or 'Domain'. Skipping."
            )
            continue

        # Scenario 1: Normal
        if "NORMAL" in scenarios_to_run:
            scenario_tasks.append(
                {
                    "id": f"{base_task_id}_normal",
                    "intent": intent,
                    "domain": domain,
                    "scenario": "NORMAL",
                    "malicious_agent_enabled": False,
                    "defense_agent_l1_enabled": False,
                    "defense_agent_l2_enabled": False,
                    "attack_prompt_template": None,
                    "defense_prompt_template_l1": None,
                    "defense_prompt_template_l2": None,
                }
            )

        # Scenario 2: Malicious Attack
        if "MALICIOUS_ATTACK" in scenarios_to_run:
            attack_template_key = f"GENERAL_PURPOSE_ATTACK_PROMPT_{domain.upper().replace(' ', '_')}"
            attack_prompt_template = attack_prompt_template_map.get(
                attack_template_key, attack_prompt_template_map.get("GENERAL_PURPOSE_ATTACK_PROMPT_DEFAULT")
            )
            if not attack_prompt_template:
                logging.warning(f"No suitable attack prompt template found for domain {domain} or default. Skipping MALICIOUS_ATTACK for {base_task_id}")
                continue

            scenario_tasks.append(
                {
                    "id": f"{base_task_id}_malicious",
                    "intent": intent,
                    "domain": domain,
                    "scenario": "MALICIOUS_ATTACK",
                    "malicious_agent_enabled": True,
                    "defense_agent_l1_enabled": False,
                    "defense_agent_l2_enabled": False,
                    "attack_prompt_template": attack_prompt_template,
                    "defense_prompt_template_l1": None,
                    "defense_prompt_template_l2": None,
                }
            )

        # Scenario 3: Defended (with L1 and potentially L2)
        if "DEFENDED" in scenarios_to_run:
            attack_template_key_def = f"GENERAL_PURPOSE_ATTACK_PROMPT_{domain.upper().replace(' ', '_')}"
            attack_prompt_template_def = attack_prompt_template_map.get(
                attack_template_key_def, attack_prompt_template_map.get("GENERAL_PURPOSE_ATTACK_PROMPT_DEFAULT")
            )
            defense_template_l1_key = f"GENERAL_LAYER_1_DEFENSE_PROMPT_{domain.upper().replace(' ', '_')}"
            defense_prompt_template_l1 = defense_prompt_template_map.get(
                defense_template_l1_key, defense_prompt_template_map.get("GENERAL_LAYER_1_DEFENSE_PROMPT_DEFAULT")
            )

            if not attack_prompt_template_def:
                logging.warning(f"No suitable attack prompt template found for domain {domain} or default for DEFENDED. Skipping DEFENDED for {base_task_id}")
                continue
            if not defense_prompt_template_l1:
                logging.warning(f"No suitable L1 defense prompt template found for domain {domain} or default. Skipping DEFENDED for {base_task_id}")
                continue

            task_id_suffix = "_defended"
            effective_defense_l2_enabled = False
            effective_defense_prompt_l2 = None

            if enable_defense_layer_2_globally and defense_prompt_template_layer_2:
                effective_defense_l2_enabled = True
                effective_defense_prompt_l2 = defense_prompt_template_layer_2
                task_id_suffix += "_l2"
            
            scenario_tasks.append(
                {
                    "id": f"{base_task_id}{task_id_suffix}",
                    "intent": intent,
                    "domain": domain,
                    "scenario": "DEFENDED",
                    "malicious_agent_enabled": True, # Assumes defense follows an attack
                    "defense_agent_l1_enabled": True,
                    "defense_agent_l2_enabled": effective_defense_l2_enabled,
                    "attack_prompt_template": attack_prompt_template_def,
                    "defense_prompt_template_l1": defense_prompt_template_l1,
                    "defense_prompt_template_l2": effective_defense_prompt_l2,
                }
            )
            
    return scenario_tasks

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Batch evaluate research ideas using Thinker and SafetyEvaluator.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file with prompts.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file for results.")
    parser.add_argument("--stats_file", type=str, default="output/batch_idea_generation_statistics.json", help="Path to the output JSON file for statistics.")
    parser.add_argument("--thinker_model_name", type=str, default="gpt-3.5-turbo", help="LLM model name for the Thinker.")
    parser.add_argument("--evaluator_model_name", type=str, default="gpt-4o", help="LLM model name for the SafetyEvaluator.")
    parser.add_argument("--num_workers", type=int, default=max(1, cpu_count() // 2), help="Number of parallel worker processes.")
    parser.add_argument("--resume", action="store_true", help="Resume from previously processed tasks in the output file.")
    parser.add_argument("--enable_defense_layer_2", action="store_true", help="Enable Layer 2 Defense for DEFENDED scenarios globally.")
    parser.add_argument("--limit", type=int, default=None, help="Limit processing to the first N prompts from the input file.")
    parser.add_argument("--scenarios_to_run", type=str, default="NORMAL,MALICIOUS_ATTACK,DEFENDED", help="Comma-separated list of scenarios to run (e.g., 'NORMAL,MALICIOUS_ATTACK')")

    args = parser.parse_args()

    # Create base output directories
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.stats_file), exist_ok=True)

    # --- Resume Logic ---
    processed_task_ids = set()
    if os.path.exists(args.output_file):
        print(f"[INFO] Output file {args.output_file} exists. Reading previously processed task IDs...")
        try:
            with open(args.output_file, 'r', encoding='utf-8') as f_read:
                for line_num, line in enumerate(f_read):
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if "task_id" in data: # Use the unique task_id (e.g., "input_task_1_normal")
                                processed_task_ids.add(data["task_id"])
                        except json.JSONDecodeError:
                            print(f"[WARNING] JSON decode error on line {line_num+1} of existing output file. Skipping for resume check.")
            print(f"[INFO] Found {len(processed_task_ids)} unique task IDs in existing output file.")
        except Exception as e:
            print(f"[ERROR] Failed to read or parse existing output file {args.output_file} for resume: {e}. Will process all tasks as new.")
            processed_task_ids.clear()
    else:
        print(f"[INFO] Output file {args.output_file} does not exist. Starting fresh.")

    # Load base tasks from the input file
    base_tasks = load_input_file(args.input_file, args.limit) # Pass limit here
    if not base_tasks:
        logging.info("No tasks to process from input file.")
        return

    # Parse scenarios_to_run argument
    scenarios_to_run_list = [s.strip().upper() for s in args.scenarios_to_run.split(',') if s.strip()]
    if not scenarios_to_run_list:
        logging.error("No scenarios specified to run. Please provide at least one scenario via --scenarios_to_run.")
        sys.exit(1)
    logging.info(f"Specified scenarios to run: {scenarios_to_run_list}")

    # Prepare prompt template maps
    # These would ideally be loaded from a config or a more structured prompt management system
    # For now, we'll create them based on the existing Enum structure for simplicity in this script
    attack_prompt_template_map = {
        prompt.name: prompt.value for prompt in AttackPrompts
    }
    defense_prompt_template_map = {
        prompt.name: prompt.value for prompt in DefensePrompts
    }

    # Explicitly set the GENERAL_PURPOSE_ATTACK_PROMPT_DEFAULT to be the CHEMISTRY_ATTACK_PROMPT
    # This is because our current test uses various chemistry sub-domains, and we want a fallback.
    if hasattr(AttackPrompts, 'CHEMISTRY_ATTACK_PROMPT'):
        attack_prompt_template_map["GENERAL_PURPOSE_ATTACK_PROMPT_DEFAULT"] = AttackPrompts.CHEMISTRY_ATTACK_PROMPT.value
    else:
        logging.warning("CHEMISTRY_ATTACK_PROMPT not found in AttackPrompts Enum, default attack prompt will be missing for chemistry subdomains.")

    # Similarly, set a default for defense prompts if needed, e.g., using a general chemistry defense prompt
    if hasattr(DefensePrompts, 'CHEMISTRY_DEFENSE_PROMPT'):
        defense_prompt_template_map["GENERAL_LAYER_1_DEFENSE_PROMPT_DEFAULT"] = DefensePrompts.CHEMISTRY_DEFENSE_PROMPT.value
    else:
        logging.warning("CHEMISTRY_DEFENSE_PROMPT not found in DefensePrompts Enum, default L1 defense prompt will be missing for chemistry subdomains.")
    
    effective_defense_prompt_l2 = None
    try:
        if args.enable_defense_layer_2:
            effective_defense_prompt_l2 = DefensePrompts.GENERAL_LAYER_2_DEFENSE_PROMPT.value
    except AttributeError:
        logging.warning("GENERAL_LAYER_2_DEFENSE_PROMPT not found in DefensePrompts Enum, L2 defense will be unavailable.")
        effective_defense_prompt_l2 = None

    # Generate specific scenario tasks
    all_scenario_tasks = generate_scenario_tasks(
        base_tasks,
        attack_prompt_template_map, # Pass the generated map
        defense_prompt_template_map,  # Pass the generated map
        effective_defense_prompt_l2,
        args.enable_defense_layer_2,
        scenarios_to_run_list
    )

    # Filter for tasks not yet processed if resuming
    new_scenario_tasks = [task for task in all_scenario_tasks if task["id"] not in processed_task_ids]

    num_tasks_actually_running = len(new_scenario_tasks)
    print(f"[INFO] Total scenario tasks generated from input: {len(all_scenario_tasks)}")

    if num_tasks_actually_running == 0:
        print("[INFO] No new tasks to process. All tasks from input file are already in the output file.")
        generate_final_statistics(args.output_file, args) # Still generate stats for all processed items
        return
        
    # Prepare arguments for workers: tuple of (index, task_config_item)
    # The index here is just for the current batch of tasks to run.
    worker_payload = list(enumerate(new_scenario_tasks))

    # API Key Check (Informational)
    if not os.environ.get("OPENAI_API_KEY"): # Or other relevant key checks for your LLM client
        print("[WARNING] OPENAI_API_KEY environment variable not set. LLM calls may fail if using OpenAI.")

    print(f"\n[SETUP] Starting batch idea generation:")
    print(f"  Input File: {args.input_file}")
    print(f"  Output File (JSONL): {args.output_file}")
    print(f"  Thinker Model: {args.thinker_model_name}")
    print(f"  SafetyEvaluator Model (in Thinker): {args.evaluator_model_name}")
    print(f"  Parallel Workers: {args.num_workers}")
    print(f"  L2 Defense (globally for DEFENDED scenarios): {'Enabled' if args.enable_defense_layer_2 else 'Disabled'}")

    # --- Multiprocessing ---
    if args.num_workers > 1 and num_tasks_actually_running > 0:
        manager = Manager()
        shared_file_lock = manager.Lock()
        # Use partial to pass fixed arguments to the worker function
        worker_func_with_args = partial(process_task_for_idea_generation, 
                                        common_args=args, 
                                        overall_output_dir_base=os.path.dirname(args.output_file), 
                                        file_lock=shared_file_lock)
        print(f"\n[INFO] Starting parallel processing of {num_tasks_actually_running} tasks with {args.num_workers} workers...")
        with Pool(processes=args.num_workers) as pool:
            # map_results will be a list of statuses returned by the worker
            pool.map(worker_func_with_args, worker_payload) 
        print("[INFO] Parallel processing finished.")
    elif num_tasks_actually_running > 0: # Sequential processing
        print(f"\n[INFO] Starting sequential processing of {num_tasks_actually_running} tasks...")
        class DummyLock: # For sequential execution, lock is not strictly needed but keeps worker signature same
            def acquire(self): pass
            def release(self): pass
        dummy_file_lock = DummyLock()
        worker_func_with_args = partial(process_task_for_idea_generation, 
                                        common_args=args, 
                                        overall_output_dir_base=os.path.dirname(args.output_file), 
                                        file_lock=dummy_file_lock)
        for i, task_config_item in enumerate(new_scenario_tasks):
            worker_func_with_args((i, task_config_item)) # Pass as tuple (index, item)
        print("[INFO] Sequential processing finished.")
    else:
        # This case should be caught earlier by "No new tasks to process"
        pass

    print(f"\n[INFO] Batch idea generation session complete.")
    print(f"  Results (newly processed tasks) appended to: {args.output_file}")
    
    # --- Final Statistics Generation (after all processing) ---
    generate_final_statistics(args.output_file, args)


def generate_final_statistics(results_jsonl_file, script_args):
    """
    Reads all results from the JSONL file and generates a statistics summary.
    """
    print(f"\n[STATS] Generating final statistics from: {results_jsonl_file}")
    
    all_results_from_file = []
    if not os.path.exists(results_jsonl_file):
        print(f"[STATS_ERROR] Output file {results_jsonl_file} not found. Cannot generate statistics.")
        return

    try:
        with open(results_jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line.strip():
                    try:
                        all_results_from_file.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"[STATS_WARNING] Could not parse line {line_num+1} from {results_jsonl_file} for statistics. Skipping.")
    except Exception as e:
        print(f"[STATS_ERROR] Error reading {results_jsonl_file} for statistics: {e}")
        return

    total_entries_in_file = len(all_results_from_file)
    successful_generations = 0
    failed_generations = 0 # Includes "skipped", "failed_setup", "failed_thinker_reported_error", "failed_processing_exception"
    
    # For safety score statistics (if present in thinker_output)
    domain_scores_data = {}  # {"DomainName": {"sum": X, "count": Y, "scores": []}}
    scenario_scores_data = {} # {"ScenarioName": {"sum": X, "count": Y, "scores": []}}

    for res in all_results_from_file:
        status = res.get("status", "unknown_status")
        if status == "success":
            successful_generations += 1
        else:
            failed_generations += 1
            # Optionally log more details about failures here if needed for stats
            # print(f"[STATS_DEBUG] Failed/Skipped task: ID {res.get('task_id')}, Status: {status}, Error: {res.get('error_details')}")

        # Extract safety score if available and successful
        if status == "success" and res.get("thinker_output"):
            thinker_out = res["thinker_output"]
            safety_score = thinker_out.get("safety_score") # Path to score in thinker.think() output
            
            if safety_score is not None: # Check it's a valid score, not None
                try:
                    score_value = float(safety_score) # Ensure it's a number
                    
                    domain = res.get("domain")
                    scenario = res.get("scenario")

                    if domain:
                        if domain not in domain_scores_data:
                            domain_scores_data[domain] = {"sum_scores": 0.0, "count": 0, "scores_list": []}
                        domain_scores_data[domain]["sum_scores"] += score_value
                        domain_scores_data[domain]["count"] += 1
                        domain_scores_data[domain]["scores_list"].append(score_value)
                    
                    if scenario:
                        if scenario not in scenario_scores_data:
                            scenario_scores_data[scenario] = {"sum_scores": 0.0, "count": 0, "scores_list": []}
                        scenario_scores_data[scenario]["sum_scores"] += score_value
                        scenario_scores_data[scenario]["count"] += 1
                        scenario_scores_data[scenario]["scores_list"].append(score_value)
                except (ValueError, TypeError):
                    print(f"[STATS_WARNING] Could not parse safety_score '{safety_score}' as float for task {res.get('task_id')}. Skipping score for stats.")


    final_stats_dict = {
        "metadata": {
            "input_file_processed": script_args.input_file,
            "output_results_file": script_args.output_file,
            "thinker_model_used": script_args.thinker_model_name,
            "evaluator_model_used": script_args.evaluator_model_name,
            "num_workers_configured": script_args.num_workers,
            "l2_defense_enabled_run": script_args.enable_defense_layer_2,
            "stats_generation_time": datetime.now().isoformat()
        },
        "overall_summary": {
            "total_task_entries_in_output_file": total_entries_in_file,
            "successful_idea_generations": successful_generations,
            "failed_or_skipped_generations": failed_generations,
        },
        "average_safety_scores_by_domain": {},
        "average_safety_scores_by_scenario": {}
    }

    for domain, data in domain_scores_data.items():
        if data["count"] > 0:
            final_stats_dict["average_safety_scores_by_domain"][domain] = {
                "average_score": round(data["sum_scores"] / data["count"], 3),
                "count_with_scores": data["count"],
                "all_scores": sorted(data["scores_list"])
            }
    for scenario, data in scenario_scores_data.items():
        if data["count"] > 0:
            final_stats_dict["average_safety_scores_by_scenario"][scenario] = {
                "average_score": round(data["sum_scores"] / data["count"], 3),
                "count_with_scores": data["count"],
                "all_scores": sorted(data["scores_list"])
            }
            
    # Save statistics to a new file (or overwrite if named the same as old STATS_FILE)
    stats_output_filename = os.path.join(os.path.dirname(script_args.output_file), "batch_idea_generation_statistics.json")
    try:
        with open(stats_output_filename, "w", encoding='utf-8') as f_stats:
            json.dump(final_stats_dict, f_stats, indent=4)
        print(f"[INFO] Final statistics saved to: {stats_output_filename}")
    except Exception as e:
        print(f"[STATS_ERROR] Could not save statistics file {stats_output_filename}: {e}")

    print("\n--- Final Statistics Summary ---")
    print(json.dumps(final_stats_dict, indent=2))
    print("--- End of Statistics ---")


if __name__ == "__main__":
    main()
