#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script for running experiments with TinyScientist using the ReAct approach.

This script demonstrates how to use TinyScientist's react_experiment method
which leverages Reasoning and Acting (ReAct) to perform domain-specific
experiments using specialized tools.
"""

import argparse
import os
import random
from pathlib import Path

from tiny_scientist import TinyScientist

# Domain-specific intents
DOMAIN_INTENTS = {
    "chemistry": [
        "Investigate the solubility of NaCl in water and ethanol at different temperatures",
        "Analyze the reaction kinetics of hydrogen peroxide decomposition with different catalysts",
        "Study the effect of pH on the rate of enzyme-catalyzed reactions",
        "Compare the thermal stability of different organic compounds using differential scanning calorimetry",
        "Investigate the relationship between molecular structure and boiling point in homologous series"
    ],
    "physics": [
        "Compare the relationship between thermal conductivity and electrical resistivity of different materials",
        "Investigate the relationship between pendulum length and oscillation period",
        "Study the effect of surface area on terminal velocity of falling objects",
        "Analyze the relationship between spring constant and oscillation frequency",
        "Investigate the relationship between angle of incidence and reflection in optics"
    ],
    "biology": [
        "Study the effect of different light wavelengths on plant growth",
        "Investigate the relationship between enzyme concentration and reaction rate",
        "Analyze the impact of temperature on bacterial growth rates",
        "Study the effect of different nutrient concentrations on cell culture growth",
        "Investigate the relationship between DNA concentration and PCR amplification efficiency"
    ],
    "material_science": [
        "Compare the mechanical properties of different polymer composites",
        "Investigate the relationship between grain size and material strength in metals",
        "Study the effect of heat treatment on material hardness",
        "Analyze the relationship between porosity and material density",
        "Investigate the impact of different alloying elements on material properties"
    ],
    "medical_science": [
        "Study the effect of different drug concentrations on cell viability",
        "Investigate the relationship between exercise intensity and heart rate",
        "Analyze the impact of different nutrients on cell growth",
        "Study the effect of temperature on enzyme activity in biological systems",
        "Investigate the relationship between blood pressure and various physiological factors"
    ],
    "information_science": [
        "Compare the performance of different compression algorithms on various data types",
        "Investigate the relationship between data size and processing time in different algorithms",
        "Study the effect of different encoding schemes on data transmission efficiency",
        "Analyze the relationship between network latency and packet size",
        "Investigate the impact of different data structures on search algorithm performance"
    ],
    "general": [
        "Explore the performance of simple machine learning models on small datasets",
        "Investigate the relationship between experimental variables using statistical analysis",
        "Study the effect of different measurement techniques on data accuracy",
        "Analyze the relationship between sample size and statistical significance",
        "Investigate the impact of different experimental conditions on measurement precision"
    ]
}

def get_random_intent(domain: str) -> str:
    """Get a random intent for the specified domain."""
    if domain not in DOMAIN_INTENTS:
        raise ValueError(f"Unknown domain: {domain}")
    return random.choice(DOMAIN_INTENTS[domain])

def main():
    parser = argparse.ArgumentParser(description="Run experiments using TinyScientist with ReAct")
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt-4o", 
        help="Specify the LLM model to use (e.g., gpt-4o, claude-3.5-sonnet)"
    )
    parser.add_argument(
        "--domain", 
        type=str, 
        default="general", 
        choices=list(DOMAIN_INTENTS.keys()),
        help="Specify the experiment domain"
    )
    parser.add_argument(
        "--intent", 
        type=str, 
        default=None,
        help="Experiment intent description (e.g., 'Chemistry experiment: measure NaCl solubility in different solvents')"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./output", 
        help="Output directory path"
    )
    parser.add_argument(
        "--max-iterations", 
        type=int, 
        default=10, 
        help="Maximum number of ReAct iterations"
    )
    parser.add_argument(
        "--template", 
        type=str, 
        default="acl", 
        choices=["acl", "iclr"],
        help="Paper template format"
    )
    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Default experiment intent (if not provided)
    if args.intent is None:
        args.intent = get_random_intent(args.domain)
    
    print(f"🚀 Using {args.model} model for a {args.domain} experiment: {args.intent}")
    
    # Initialize TinyScientist
    scientist = TinyScientist(
        model=args.model,
        output_dir=output_dir,
        template=args.template
    )
    
    # Step 1: Generate research idea
    # This uses the thinker component to develop a structured research plan
    print("Generating research idea...")
    idea = scientist.think(intent=args.intent)
    
    # Step 2: Execute experiment using react_experiment method
    # The ReactExperimenter will:
    # - Load domain-specific tools (chemistry, physics, or general)
    # - Run a ReAct loop where the LLM reasons and acts using available tools
    # - Log the experiment process and results
    print(f"Executing {args.domain} experiment with ReAct...")
    status, experiment_dir = scientist.react_experiment(
        idea=idea, 
        domain=args.domain,
        max_iterations=args.max_iterations
    )
    
    # If experiment successful, generate research paper
    if status:
        # Step 3: Write paper
        # This uses the experiment logs and results to generate a structured paper
        print("Writing research paper...")
        pdf_path = scientist.write(idea=idea, experiment_dir=experiment_dir)
        
        # Step 4: Review paper (optional)
        # This analyzes the paper for quality and scientific merit
        print("Reviewing paper...")
        review = scientist.review(pdf_path=pdf_path)
        
        print(f"✅ Experiment workflow complete!")
        print(f"📄 Paper saved at: {pdf_path}")
    else:
        print("❌ Experiment failed to complete successfully. Cannot generate paper.")


if __name__ == "__main__":
    main() 