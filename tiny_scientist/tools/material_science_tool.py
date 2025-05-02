import json
import math
from typing import Any, Dict, List, Optional, Union

from ..tool import BaseTool
from ..utils.llm import create_client, get_response_from_llm


class MaterialPropertiesTool(BaseTool):
    """Tool for material properties analysis and calculations."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the tool with a language model for material analysis."""
        self.client, self.model = create_client(model)
        # Common material properties database
        self.materials = {
            "steel": {
                "density": 7850,  # kg/m³
                "youngs_modulus": 200e9,  # Pa
                "poissons_ratio": 0.3,
                "thermal_conductivity": 50,  # W/(m·K)
                "specific_heat": 450,  # J/(kg·K)
                "thermal_expansion": 12e-6,  # 1/K
                "electrical_resistivity": 1.7e-7,  # Ω·m
                "melting_point": 1370,  # °C
                "yield_strength": 250e6,  # Pa
                "ultimate_strength": 400e6  # Pa
            },
            "aluminum": {
                "density": 2700,
                "youngs_modulus": 69e9,
                "poissons_ratio": 0.33,
                "thermal_conductivity": 237,
                "specific_heat": 900,
                "thermal_expansion": 23e-6,
                "electrical_resistivity": 2.7e-8,
                "melting_point": 660,
                "yield_strength": 70e6,
                "ultimate_strength": 110e6
            },
            "copper": {
                "density": 8960,
                "youngs_modulus": 110e9,
                "poissons_ratio": 0.34,
                "thermal_conductivity": 401,
                "specific_heat": 385,
                "thermal_expansion": 17e-6,
                "electrical_resistivity": 1.7e-8,
                "melting_point": 1085,
                "yield_strength": 33e6,
                "ultimate_strength": 210e6
            },
            "titanium": {
                "density": 4500,
                "youngs_modulus": 116e9,
                "poissons_ratio": 0.32,
                "thermal_conductivity": 21.9,
                "specific_heat": 523,
                "thermal_expansion": 8.6e-6,
                "electrical_resistivity": 4.2e-7,
                "melting_point": 1668,
                "yield_strength": 140e6,
                "ultimate_strength": 240e6
            }
        }
    
    def safety_detect(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Check if the material query is safe to process."""
        # Check for potentially dangerous materials
        dangerous_patterns = [
            "radioactive", "toxic", "explosive", "flammable",
            "corrosive", "hazardous", "carcinogenic"
        ]
        
        for key, value in inputs.items():
            if isinstance(value, str):
                for pattern in dangerous_patterns:
                    if pattern in value.lower():
                        return {
                            "safe": False,
                            "reason": f"Potentially dangerous material detected: {pattern}",
                            "allowed": False
                        }
        
        return {"safe": True, "reason": "No safety concerns detected", "allowed": True}
    
    def run(
        self,
        material: str,
        analysis_type: str = "basic",
        temperature: Optional[float] = None,
        pressure: Optional[float] = None,
        stress: Optional[float] = None,
        strain: Optional[float] = None,
        dimensions: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Analyze material properties and perform calculations.
        
        Args:
            material: Material name
            analysis_type: Type of analysis ("basic", "mechanical", "thermal", etc.)
            temperature: Temperature in °C
            pressure: Pressure in Pa
            stress: Stress in Pa
            strain: Strain (dimensionless)
            dimensions: Dictionary of dimensions (length, width, height) in meters
            
        Returns:
            Dictionary with analysis results
        """
        # Check safety first
        inputs = {"material": material}
        safety_result = self.safety_detect(inputs)
        if not safety_result.get("allowed", True):
            return {
                "success": False,
                "error": "Safety check failed",
                "details": safety_result
            }
        
        # Normalize material name
        material = material.lower()
        result = {"success": True, "analysis_type": analysis_type}
        
        if material not in self.materials:
            return {
                "success": False,
                "error": f"Material '{material}' not found in database"
            }
        
        mat_props = self.materials[material]
        
        # Basic properties
        if analysis_type.lower() in ["basic", "all"]:
            result["density"] = f"{mat_props['density']} kg/m³"
            result["youngs_modulus"] = f"{mat_props['youngs_modulus']:.2e} Pa"
            result["poissons_ratio"] = mat_props["poissons_ratio"]
            result["thermal_conductivity"] = f"{mat_props['thermal_conductivity']} W/(m·K)"
            result["specific_heat"] = f"{mat_props['specific_heat']} J/(kg·K)"
        
        # Mechanical properties
        if analysis_type.lower() in ["mechanical", "all"]:
            if stress is not None and strain is not None:
                result["youngs_modulus_calculated"] = f"{stress/strain:.2e} Pa"
            
            if dimensions:
                volume = dimensions.get("length", 0) * dimensions.get("width", 0) * dimensions.get("height", 0)
                mass = volume * mat_props["density"]
                result["mass"] = f"{mass:.2f} kg"
                result["weight"] = f"{mass * 9.81:.2f} N"
            
            result["yield_strength"] = f"{mat_props['yield_strength']:.2e} Pa"
            result["ultimate_strength"] = f"{mat_props['ultimate_strength']:.2e} Pa"
        
        # Thermal properties
        if analysis_type.lower() in ["thermal", "all"]:
            if temperature is not None:
                result["thermal_conductivity"] = f"{mat_props['thermal_conductivity']} W/(m·K)"
                result["specific_heat"] = f"{mat_props['specific_heat']} J/(kg·K)"
                result["thermal_expansion"] = f"{mat_props['thermal_expansion']} 1/K"
                result["melting_point"] = f"{mat_props['melting_point']}°C"
            
            if dimensions and temperature is not None:
                volume = dimensions.get("length", 0) * dimensions.get("width", 0) * dimensions.get("height", 0)
                mass = volume * mat_props["density"]
                heat_capacity = mass * mat_props["specific_heat"]
                result["heat_capacity"] = f"{heat_capacity:.2f} J/K"
        
        # Electrical properties
        if analysis_type.lower() in ["electrical", "all"]:
            result["electrical_resistivity"] = f"{mat_props['electrical_resistivity']:.2e} Ω·m"
            result["electrical_conductivity"] = f"{1/mat_props['electrical_resistivity']:.2e} S/m"
        
        return result


class MaterialAnalysisTool(BaseTool):
    """Tool for material analysis and characterization."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the tool with a language model for material analysis."""
        self.client, self.model = create_client(model)
        # Common material analysis parameters
        self.constants = {
            "crystal_structures": {
                "bcc": {"atoms_per_cell": 2, "packing_factor": 0.68},
                "fcc": {"atoms_per_cell": 4, "packing_factor": 0.74},
                "hcp": {"atoms_per_cell": 6, "packing_factor": 0.74},
                "sc": {"atoms_per_cell": 1, "packing_factor": 0.52}
            },
            "defect_types": [
                "vacancy", "interstitial", "substitutional",
                "dislocation", "grain_boundary", "twin_boundary"
            ],
            "phase_transitions": {
                "solid-solid": ["allotropic", "polymorphic", "martensitic"],
                "solid-liquid": ["melting", "freezing"],
                "solid-gas": ["sublimation", "deposition"],
                "liquid-gas": ["vaporization", "condensation"]
            }
        }
    
    def safety_detect(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Check if the material analysis query is safe to process."""
        # Check for potentially dangerous materials or conditions
        dangerous_patterns = [
            "radioactive", "toxic", "explosive", "flammable",
            "corrosive", "hazardous", "carcinogenic", "high_pressure",
            "high_temperature", "vacuum", "cryogenic"
        ]
        
        for key, value in inputs.items():
            if isinstance(value, str):
                for pattern in dangerous_patterns:
                    if pattern in value.lower():
                        return {
                            "safe": False,
                            "reason": f"Potentially dangerous condition detected: {pattern}",
                            "allowed": False
                        }
        
        return {"safe": True, "reason": "No safety concerns detected", "allowed": True}
    
    def run(
        self,
        material: str,
        analysis_type: str = "basic",
        crystal_structure: Optional[str] = None,
        temperature: Optional[float] = None,
        pressure: Optional[float] = None,
        composition: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Analyze material structure and properties.
        
        Args:
            material: Material name
            analysis_type: Type of analysis ("basic", "crystal", "defects", etc.)
            crystal_structure: Crystal structure type
            temperature: Temperature in °C
            pressure: Pressure in Pa
            composition: Dictionary of element percentages
            
        Returns:
            Dictionary with analysis results
        """
        # Check safety first
        inputs = {
            "material": material,
            "temperature": str(temperature) if temperature else "",
            "pressure": str(pressure) if pressure else ""
        }
        safety_result = self.safety_detect(inputs)
        if not safety_result.get("allowed", True):
            return {
                "success": False,
                "error": "Safety check failed",
                "details": safety_result
            }
        
        result = {"success": True, "analysis_type": analysis_type}
        
        # Crystal structure analysis
        if analysis_type.lower() in ["crystal", "all"] and crystal_structure:
            if crystal_structure.lower() in self.constants["crystal_structures"]:
                structure = self.constants["crystal_structures"][crystal_structure.lower()]
                result["atoms_per_unit_cell"] = structure["atoms_per_cell"]
                result["atomic_packing_factor"] = structure["packing_factor"]
                result["coordination_number"] = self._calculate_coordination_number(crystal_structure)
        
        # Defect analysis
        if analysis_type.lower() in ["defects", "all"]:
            result["possible_defects"] = self._analyze_possible_defects(material, crystal_structure)
            if temperature is not None:
                result["defect_concentration"] = self._calculate_defect_concentration(temperature)
        
        # Phase analysis
        if analysis_type.lower() in ["phase", "all"]:
            if temperature is not None and pressure is not None:
                result["phase_stability"] = self._analyze_phase_stability(temperature, pressure)
                result["possible_transitions"] = self._analyze_phase_transitions(temperature, pressure)
        
        # Composition analysis
        if analysis_type.lower() in ["composition", "all"] and composition:
            result["composition_analysis"] = self._analyze_composition(composition)
            result["stoichiometry"] = self._calculate_stoichiometry(composition)
        
        return result
    
    def _calculate_coordination_number(self, crystal_structure: str) -> int:
        """Calculate coordination number for given crystal structure."""
        if crystal_structure.lower() == "bcc":
            return 8
        elif crystal_structure.lower() == "fcc":
            return 12
        elif crystal_structure.lower() == "hcp":
            return 12
        elif crystal_structure.lower() == "sc":
            return 6
        return 0
    
    def _analyze_possible_defects(self, material: str, crystal_structure: Optional[str]) -> List[str]:
        """Analyze possible defects in the material."""
        defects = []
        if crystal_structure:
            defects.extend(["vacancy", "interstitial"])
            if crystal_structure.lower() in ["fcc", "bcc"]:
                defects.append("dislocation")
            if crystal_structure.lower() in ["hcp"]:
                defects.append("twin_boundary")
        defects.append("grain_boundary")
        return defects
    
    def _calculate_defect_concentration(self, temperature: float) -> float:
        """Calculate equilibrium defect concentration at given temperature."""
        # Simplified Arrhenius equation
        activation_energy = 1.0  # eV
        boltzmann_constant = 8.617e-5  # eV/K
        return math.exp(-activation_energy/(boltzmann_constant*(temperature + 273.15)))
    
    def _analyze_phase_stability(self, temperature: float, pressure: float) -> str:
        """Analyze phase stability at given conditions."""
        if temperature > 1000 and pressure > 1e9:
            return "unstable"
        elif temperature > 500 or pressure > 1e8:
            return "metastable"
        return "stable"
    
    def _analyze_phase_transitions(self, temperature: float, pressure: float) -> List[str]:
        """Analyze possible phase transitions at given conditions."""
        transitions = []
        if temperature > 1000:
            transitions.extend(self.constants["phase_transitions"]["solid-liquid"])
        if pressure > 1e9:
            transitions.extend(self.constants["phase_transitions"]["solid-solid"])
        return transitions
    
    def _analyze_composition(self, composition: Dict[str, float]) -> Dict[str, Any]:
        """Analyze material composition."""
        total = sum(composition.values())
        normalized = {k: v/total for k, v in composition.items()}
        return {
            "elements": normalized,
            "major_element": max(normalized.items(), key=lambda x: x[1])[0],
            "minor_elements": [k for k, v in normalized.items() if v < 0.1]
        }
    
    def _calculate_stoichiometry(self, composition: Dict[str, float]) -> Dict[str, float]:
        """Calculate stoichiometric ratios."""
        # Simplified calculation assuming atomic weights are equal
        total = sum(composition.values())
        return {k: v/total for k, v in composition.items()} 