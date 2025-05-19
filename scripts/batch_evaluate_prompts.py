import argparse
import json
import os
from multiprocessing import Pool, cpu_count
# Updated imports based on class locations and new utils
from tiny_scientist import Thinker, SafetyEvaluator # ResearchIdea might not be directly used here now
from tiny_scientist.prompts.prompts import AttackPrompts, DefensePrompts # ScientistPrompts not used directly by script
from tiny_scientist.utils.llm import create_client # For creating the LLM client

# Ensure the output directory exists
os.makedirs("output/batch_evaluation_jsonl", exist_ok=True)
OUTPUT_FILE = "output/batch_evaluation_jsonl/batch_results.jsonl"
STATS_FILE = "output/batch_evaluation_jsonl/batch_statistics.json"

# Default model names
DEFAULT_THINKER_MODEL = "gpt-3.5-turbo"
DEFAULT_EVALUATOR_MODEL = "gpt-4o"

# Global LLM client and model name - to be initialized in main()
# This avoids re-creating clients in each process if possible, 
# but multiprocessing with some LLM clients can be tricky.
# For now, let's initialize it per process in `process_prompt` for safety with multiprocessing.
# If LLMAPIFactory handles shared clients or if create_client is lightweight, this can be optimized.

# Placeholder for loading prompts (Unchanged from previous version)
def load_prompts():
    """
    Loads prompts for different scenarios and scientific domains.
    Reads the first 10 entries from 'AAA Dataset/material.json' for Material Science.
    """
    output_prompts = []
    data_file_path = "AAA Dataset/material.json" # Path to your data file
    domain = "Material Science"

    try:
        with open(data_file_path, 'r') as f:
            all_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The data file {data_file_path} was not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {data_file_path}.")
        return []

    # Take the first 10 entries, or fewer if the file has less than 10
    for i, item in enumerate(all_data[:10]): # Added [:10] slice to limit to first 10
        intent = item.get("Prompt")
        task_name = item.get("Task", f"material_task_{i+1}") # Use Task as ID, or generate one

        if not intent:
            print(f"Warning: Item {i} in {data_file_path} is missing 'Prompt' field. Skipping.")
            continue

        # 1. NORMAL scenario
        output_prompts.append({
            "id": f"{task_name}_normal",
            "domain": domain,
            "scenario": "NORMAL",
            "intent": intent,
            "attack_prompt_template": None,
            "defense_prompt_template": None
        })

        # 2. MALICIOUS_ATTACK scenario
        output_prompts.append({
            "id": f"{task_name}_malicious",
            "domain": domain,
            "scenario": "MALICIOUS_ATTACK",
            "intent": intent,
            "attack_prompt_template": AttackPrompts.MATERIAL_SCIENCE_ATTACK_PROMPT.value,
            "defense_prompt_template": None
        })

        # 3. DEFENDED scenario
        output_prompts.append({
            "id": f"{task_name}_defended",
            "domain": domain,
            "scenario": "DEFENDED",
            "intent": intent,
            "attack_prompt_template": AttackPrompts.MATERIAL_SCIENCE_ATTACK_PROMPT.value,
            "defense_prompt_template": DefensePrompts.MATERIAL_SCIENCE_DEFENSE_PROMPT.value
        })
            
    return output_prompts

# Global args, to be set in main()
def_model_name = "gpt-4o" # Default model, can be overridden by args

def process_prompt(process_args):
    """
    Processes a single prompt item. Now accepts a tuple (prompt_item, global_args_dict)
    to allow passing model_name and other global settings if needed.
    """
    prompt_item, thinker_model_arg, evaluator_model_arg = process_args
    
    print(f"Processing PID: {os.getpid()}, Prompt ID: {prompt_item['id']} ({prompt_item['domain']} - {prompt_item['scenario']}) with Thinker: {thinker_model_arg}, Evaluator: {evaluator_model_arg}")

    # Initialize LLM client for this process
    # This ensures each process in the pool has its own client if required by the LLM library.
    # OPENAI_API_KEY should be in the environment.
    try:
        thinker_llm_client, thinker_actual_model_name = create_client(model=thinker_model_arg)
        evaluator_llm_client, evaluator_actual_model_name = create_client(model=evaluator_model_arg)
    except Exception as e:
        print(f"Error creating LLM clients: {e}")
        return { "prompt_id": prompt_item["id"], "error": f"LLM client creation failed: {e}", **prompt_item }

    # Instantiate SafetyEvaluator with the client and model name
    safety_evaluator = SafetyEvaluator(
        llm_client=evaluator_llm_client,
        model_name=evaluator_actual_model_name 
        # Uses default prompt template from safety_evaluator.py
        # evaluation_target_type and evaluation_content_header use defaults
    )

    # Configure thinker based on scenario
    enable_malicious = False
    enable_defense = False
    attack_template_str = None
    defense_template_str = None

    if prompt_item["scenario"] == "MALICIOUS_ATTACK":
        enable_malicious = True
        attack_template_str = prompt_item.get("attack_prompt_template")
    elif prompt_item["scenario"] == "DEFENDED":
        enable_malicious = True # Defense implies an attack scenario first
        enable_defense = True
        attack_template_str = prompt_item.get("attack_prompt_template")
        defense_template_str = prompt_item.get("defense_prompt_template")

    # Instantiate Thinker
    thinker = Thinker(
        llm_client=thinker_llm_client,
        model_name=thinker_actual_model_name,
        initial_research_intent=prompt_item["intent"],
        domain=prompt_item["domain"],
        safety_evaluator=safety_evaluator,
        enable_malicious_agents=enable_malicious,
        enable_defense_agent=enable_defense,
        attack_prompt_template=attack_template_str,
        defense_prompt_template=defense_template_str,
        tools=[],  # Default: no special tools for this evaluation setup
        iter_num=1, # Default: number of iterations/rounds if thinker uses it internally
        # Other Thinker params like output_dir, temperature, etc., can be set if needed
        # For temperature, the client/SafetyEvaluator/Thinker might have their own defaults or settings.
        # The LLM calls within these classes will use their respective temperature settings.
    )

    try:
        research_idea_dict = thinker.think() # Expecting a dictionary from the refactored Thinker.think()
        
        result = {
            "prompt_id": prompt_item["id"],
            "domain": prompt_item["domain"],
            "scenario": prompt_item["scenario"],
            "intent": prompt_item["intent"],
            "final_idea": research_idea_dict.get("final_idea_description"),
            "safety_score": research_idea_dict.get("safety_score"),
            "safety_justification": research_idea_dict.get("safety_justification"),
            "raw_idea_stage1": research_idea_dict.get("raw_idea_stage1"),
            "attack_influenced_idea_stage2": research_idea_dict.get("attack_influenced_idea_stage2"),
            "defended_idea_stage3": research_idea_dict.get("defended_idea_stage3"),
            "discussion_history": research_idea_dict.get("discussion_history"), # If Thinker returns it
            "error": None
        }

    except Exception as e:
        print(f"Error processing prompt ID {prompt_item['id']} in Thinker: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        result = {
            "prompt_id": prompt_item["id"],
            "domain": prompt_item["domain"],
            "scenario": prompt_item["scenario"],
            "intent": prompt_item["intent"],
            "final_idea": None, "safety_score": None, "safety_justification": None,
            "raw_idea_stage1": None, "attack_influenced_idea_stage2": None, "defended_idea_stage3": None,
            "error": str(e)
        }
    return result

def save_result(result):
    with open(OUTPUT_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Batch evaluate research prompts for safety.")
    parser.add_argument("--num_workers", type=int, default=min(4, cpu_count()), help="Number of worker processes.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of prompts to process.")
    parser.add_argument("--thinker_model_name", type=str, default=DEFAULT_THINKER_MODEL, help=f"LLM for Thinker (default: {DEFAULT_THINKER_MODEL}).")
    parser.add_argument("--evaluator_model_name", type=str, default=DEFAULT_EVALUATOR_MODEL, help=f"LLM for SafetyEvaluator (default: {DEFAULT_EVALUATOR_MODEL}).")
    # Add other relevant arguments if needed, e.g., for Thinker's iter_num, temperature, etc.

    args = parser.parse_args()

    # Ensure OPENAI_API_KEY is set (or other relevant keys for the chosen LLM provider via create_client)
    if not os.environ.get("OPENAI_API_KEY"):
        # This check is specific to OpenAI. create_client might handle other providers/auth methods.
        print("Warning: OPENAI_API_KEY environment variable not found. LLM calls may fail if using OpenAI.")

    all_prompts = load_prompts()
    prompts_to_process_items = all_prompts
    if args.limit:
        prompts_to_process_items = prompts_to_process_items[:args.limit]
    
    # Prepare arguments for pool.map - each item needs the prompt_item and global_args_dict
    map_args = [(item, args.thinker_model_name, args.evaluator_model_name) for item in prompts_to_process_items]

    if os.path.exists(OUTPUT_FILE):
        print(f"Output file {OUTPUT_FILE} exists. Removing previous results.")
        os.remove(OUTPUT_FILE)
    if os.path.exists(STATS_FILE):
        print(f"Statistics file {STATS_FILE} exists. Removing previous stats.")
        os.remove(STATS_FILE)
    
    print(f"Starting batch evaluation with {args.num_workers} worker(s) for {len(prompts_to_process_items)} prompt(s).")
    print(f"Thinker model: {args.thinker_model_name}, Evaluator model: {args.evaluator_model_name}")

    # Using 'spawn' start method for better compatibility, especially if using CUDA/GPU with some libraries.
    # Python's default on macOS and Windows from 3.8+ is 'spawn' for multiprocessing.Pool.
    # Explicitly setting can sometimes help avoid issues.
    # import multiprocessing as mp
    # ctx = mp.get_context('spawn')
    # with ctx.Pool(processes=args.num_workers) as pool:
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(process_prompt, map_args)

    total_prompts = len(results)
    successful_evaluations = 0
    failed_evaluations = 0
    domain_scores = {}
    scenario_scores = {}

    for res in results:
        save_result(res) # Save individual result
        if res.get("error") is None and res.get("safety_score") is not None:
            successful_evaluations += 1
            domain = res["domain"]
            if domain not in domain_scores: domain_scores[domain] = {"sum_scores": 0, "count": 0, "scores": []}
            domain_scores[domain]["sum_scores"] += res["safety_score"]
            domain_scores[domain]["count"] += 1
            domain_scores[domain]["scores"].append(res["safety_score"])

            scenario = res["scenario"]
            if scenario not in scenario_scores: scenario_scores[scenario] = {"sum_scores": 0, "count": 0, "scores": []}
            scenario_scores[scenario]["sum_scores"] += res["safety_score"]
            scenario_scores[scenario]["count"] += 1
            scenario_scores[scenario]["scores"].append(res["safety_score"])
        elif res.get("error") is not None:
            failed_evaluations += 1
            print(f"Evaluation failed for prompt ID {res.get('prompt_id', 'N/A')}: {res['error']}")
        # Consider cases where score is None but no error string (should be caught by Thinker/SafetyEvaluator ideally)

    statistics = {
        "total_prompts_processed": total_prompts,
        "successful_evaluations": successful_evaluations,
        "failed_evaluations": failed_evaluations,
        "domain_average_scores": {},
        "scenario_average_scores": {},
        "settings": {
            "thinker_model_name": args.thinker_model_name, 
            "evaluator_model_name": args.evaluator_model_name, 
            "num_workers": args.num_workers, 
            "limit": args.limit
        }
    }

    for domain, data in domain_scores.items():
        if data["count"] > 0:
            statistics["domain_average_scores"][domain] = {"average_score": data["sum_scores"] / data["count"], "count": data["count"], "scores": sorted(data["scores"])}
    for scenario, data in scenario_scores.items():
        if data["count"] > 0:
            statistics["scenario_average_scores"][scenario] = {"average_score": data["sum_scores"] / data["count"], "count": data["count"], "scores": sorted(data["scores"])}

    with open(STATS_FILE, "w") as f:
        json.dump(statistics, f, indent=4)

    print(f"\nBatch evaluation complete. Results saved to {OUTPUT_FILE}")
    print(f"Statistics saved to {STATS_FILE}")
    print("\nSummary Statistics:")
    print(json.dumps(statistics, indent=4))

if __name__ == "__main__":
    main() 