"""
Example intents for different scientific domains in TinyScientist experiments.

This file contains example experiment intents for various scientific domains,
including chemistry, physics, biology, material science, medical science,
and information science. These intents can be used as templates or starting
points for running experiments with TinyScientist.
"""

# Chemistry domain intents
CHEMISTRY_INTENTS = [
    "Investigate the solubility of NaCl in water and ethanol at different temperatures",
    "Analyze the reaction kinetics of hydrogen peroxide decomposition with different catalysts",
    "Study the effect of pH on the rate of enzyme-catalyzed reactions",
    "Compare the thermal stability of different organic compounds using differential scanning calorimetry",
    "Investigate the relationship between molecular structure and boiling point in homologous series"
]

# Physics domain intents
PHYSICS_INTENTS = [
    "Compare the relationship between thermal conductivity and electrical resistivity of different materials",
    "Investigate the relationship between pendulum length and oscillation period",
    "Study the effect of surface area on terminal velocity of falling objects",
    "Analyze the relationship between spring constant and oscillation frequency",
    "Investigate the relationship between angle of incidence and reflection in optics"
]

# Biology domain intents
BIOLOGY_INTENTS = [
    "Study the effect of different light wavelengths on plant growth",
    "Investigate the relationship between enzyme concentration and reaction rate",
    "Analyze the impact of temperature on bacterial growth rates",
    "Study the effect of different nutrient concentrations on cell culture growth",
    "Investigate the relationship between DNA concentration and PCR amplification efficiency"
]

# Material Science domain intents
MATERIAL_SCIENCE_INTENTS = [
    "Compare the mechanical properties of different polymer composites",
    "Investigate the relationship between grain size and material strength in metals",
    "Study the effect of heat treatment on material hardness",
    "Analyze the relationship between porosity and material density",
    "Investigate the impact of different alloying elements on material properties"
]

# Medical Science domain intents
MEDICAL_SCIENCE_INTENTS = [
    "Study the effect of different drug concentrations on cell viability",
    "Investigate the relationship between exercise intensity and heart rate",
    "Analyze the impact of different nutrients on cell growth",
    "Study the effect of temperature on enzyme activity in biological systems",
    "Investigate the relationship between blood pressure and various physiological factors"
]

# Information Science domain intents
INFORMATION_SCIENCE_INTENTS = [
    "Compare the performance of different compression algorithms on various data types",
    "Investigate the relationship between data size and processing time in different algorithms",
    "Study the effect of different encoding schemes on data transmission efficiency",
    "Analyze the relationship between network latency and packet size",
    "Investigate the impact of different data structures on search algorithm performance"
]

# General Science intents
GENERAL_INTENTS = [
    "Explore the performance of simple machine learning models on small datasets",
    "Investigate the relationship between experimental variables using statistical analysis",
    "Study the effect of different measurement techniques on data accuracy",
    "Analyze the relationship between sample size and statistical significance",
    "Investigate the impact of different experimental conditions on measurement precision"
]

# Dictionary mapping domains to their intents
DOMAIN_INTENTS = {
    "chemistry": CHEMISTRY_INTENTS,
    "physics": PHYSICS_INTENTS,
    "biology": BIOLOGY_INTENTS,
    "material_science": MATERIAL_SCIENCE_INTENTS,
    "medical_science": MEDICAL_SCIENCE_INTENTS,
    "information_science": INFORMATION_SCIENCE_INTENTS,
    "general": GENERAL_INTENTS
}

def get_random_intent(domain: str) -> str:
    """
    Get a random intent for the specified domain.
    
    Args:
        domain: The scientific domain (e.g., "chemistry", "physics")
        
    Returns:
        A random intent string for the specified domain
    """
    import random
    if domain not in DOMAIN_INTENTS:
        raise ValueError(f"Unknown domain: {domain}")
    return random.choice(DOMAIN_INTENTS[domain])

def get_all_intents() -> Dict[str, List[str]]:
    """
    Get all intents for all domains.
    
    Returns:
        Dictionary mapping domains to their lists of intents
    """
    return DOMAIN_INTENTS 