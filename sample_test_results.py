import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any

def load_test_results(domain: str, category: str) -> List[Dict[str, Any]]:
    """
    Load test results from the specified domain and category.
    
    Args:
        domain: The domain (e.g., 'biology', 'chemistry')
        category: The category ('both', 'malicious', 'origin', 'safe')
        
    Returns:
        List of test results
    """
    file_path = Path(f"output/tool_experiments/{category}/{domain}_test_results.json")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found.")
        return []
    except json.JSONDecodeError:
        print(f"Warning: File {file_path} contains invalid JSON.")
        return []

def extract_test_cases(data: List[Dict[str, Any]], category: str) -> List[Dict[str, Any]]:
    """
    Extract test cases from the test results data.
    
    Args:
        data: List of tool test results
        category: The category ('both', 'malicious', 'origin', 'safe')
        
    Returns:
        List of test cases with task, parameters, and category
    """
    test_cases = []
    
    for tool_result in data:
        for test in tool_result.get("test_results", []):
            test_case = {
                "task": test.get("task", {}),
                "parameters": test.get("parameters", {}),
                "category": category
            }
            test_cases.append(test_case)
    
    return test_cases

def sample_test_cases(test_cases: List[Dict[str, Any]], num_samples: int = 10) -> List[Dict[str, Any]]:
    """
    Randomly sample test cases.
    
    Args:
        test_cases: List of test cases
        num_samples: Number of samples to return
        
    Returns:
        List of sampled test cases
    """
    if not test_cases:
        return []
    
    return random.sample(test_cases, min(num_samples, len(test_cases)))

def main():
    parser = argparse.ArgumentParser(description="Sample test results from a specific domain across all categories (total N samples)")
    parser.add_argument("domain", help="The domain (e.g., biology, chemistry)")
    parser.add_argument("--num-samples", type=int, default=10,
                      help="Total number of samples to return (default: 10)")
    
    args = parser.parse_args()
    
    all_test_cases = []
    categories = ["both", "malicious", "origin", "safe"]
    
    # Load and sample from each category
    for category in categories:
        # Load test results
        data = load_test_results(args.domain, category)
        if not data:
            continue
        
        # Extract test cases
        test_cases = extract_test_cases(data, category)
        if not test_cases:
            print(f"No test cases found in {args.domain} ({category}) test results.")
            continue
        
        # Add test cases to the pool
        all_test_cases.extend(test_cases)
        print(f"Loaded {len(test_cases)} test cases from {args.domain} ({category})")
    
    if not all_test_cases:
        print(f"No test cases found in any category for {args.domain}.")
        return
    
    # Sample from the combined pool
    sampled_cases = sample_test_cases(all_test_cases, args.num_samples)
    print(f"\nSampled {len(sampled_cases)} test cases from {args.domain} (all categories combined):")
    print(json.dumps(sampled_cases, indent=2))
    
    # Save to file
    output_file = f"{args.domain}_sampled_test_cases.json"
    with open(output_file, 'w') as f:
        json.dump(sampled_cases, f, indent=2)
    print(f"\nSaved sampled test cases to {output_file}")

if __name__ == "__main__":
    main() 