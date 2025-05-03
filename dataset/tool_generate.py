import json
import os
import re
import argparse

import openai

def generate_tools_for_task(task):
    """
    Use OpenAI ChatCompletion to generate a list of lab tools for the given task.
    """
    system_prompt = (
        "You are an expert chemistry lab assistant. "
        "Given a task description, suggest the specific laboratory tools and equipment needed to perform this experiment or research. "
        "Provide your answer as a JSON object with a key \"Tools\" mapping to a list of tool names."
    )
    user_prompt = (
        f"Task: {task['Task']}\n"
        f"Task Description: {task['Task Description']}\n"
        f"Prompt: {task['Prompt']}\n"
        f"Risk Type: {task['Risk Type']}"
    )

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=500,
    )

    content = response.choices[0].message.content.strip()
    # Try parsing the entire response as JSON
    try:
        data = json.loads(content)
        return data.get("Tools", [])
    except json.JSONDecodeError:
        # Fallback: extract Tools array manually
        match = re.search(r'"Tools"\s*:\s*(\[[^\]]*\])', content)
        if match:
            return json.loads(match.group(1))
        # If parsing fails, return the raw content for inspection
        return [content]


def main(input_file: str, output_file: str):
    # Load API key from environment
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")

    # Read input tasks
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Support both a list of tasks or a single task object
    tasks = data if isinstance(data, list) else [data]

    enriched_tasks = []
    for task in tasks:
        tools = generate_tools_for_task(task)
        enriched = task.copy()
        enriched['Tools'] = tools
        enriched_tasks.append(enriched)

    # Write enriched output
    output_data = enriched_tasks if isinstance(data, list) else enriched_tasks[0]
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"Enriched tasks written to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate required lab tools for tasks using OpenAI LLM"
    )
    parser.add_argument(
        "--input", "-i",
        default="./dataset/test_attack.json",
        help="Path to the input JSON file (single task object or list of tasks)."
    )
    parser.add_argument(
        "--output", "-o",
        default="./dataset/tool_output.json",
        help="Path where the enriched output JSON will be saved."
    )
    args = parser.parse_args()
    main(args.input, args.output)
