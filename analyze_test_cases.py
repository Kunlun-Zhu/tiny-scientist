import json
from collections import defaultdict

def analyze_test_cases(file_path):
    # Read the test cases
    with open(file_path, 'r') as f:
        test_cases = json.load(f)
    
    # Initialize counters for each category
    category_stats = defaultdict(lambda: {'total': 0, 'safe': 0})
    
    # Analyze each test case
    for case in test_cases:
        category = case.get('category', 'unknown')
        is_safe = case.get('safe', False)
        
        category_stats[category]['total'] += 1
        if is_safe != False:
            category_stats[category]['safe'] += 1
    
    return category_stats

if __name__ == "__main__":
    # Analyze each domain's test cases
    domains = ['biology', 'chemical', 'medical', 'physics', 'material', "is"]
    categories = ['safe', 'malicious', 'origin', 'both']
    
    # Initialize total stats
    total_stats = defaultdict(lambda: {'total': 0, 'safe': 0})
    
    print("\nAnalyzing all domains:")
    print("-" * 50)
    
    for domain in domains:
        file_path = f"{domain}_sampled_test_cases.json"
        try:
            domain_stats = analyze_test_cases(file_path)
            # Add to total stats
            for category, stats in domain_stats.items():
                total_stats[category]['total'] += stats['total']
                total_stats[category]['safe'] += stats['safe']
        except FileNotFoundError:
            print(f"No test cases found for {domain}")
    
    # Print total summary
    print("\nTotal Test Case Analysis Summary (All Domains):")
    print("-" * 50)
    for category in categories:
        stats = total_stats[category]
        print(f"{stats['total']} tasks are in {category} category and {stats['safe']} of them are safe")
    print("-" * 50) 