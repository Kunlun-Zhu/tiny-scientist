import json
import math
from typing import Any, Dict, List, Optional, Union

from ..tool import BaseTool
from ..utils.llm import create_client, get_response_from_llm


class DNATool(BaseTool):
    """Tool for DNA sequence analysis and manipulation."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the tool with a language model for DNA analysis."""
        self.client, self.model = create_client(model)
        # Common DNA-related constants and properties
        self.constants = {
            "base_pairs_per_turn": 10.5,
            "helix_rise": 0.34,  # nm per base pair
            "melting_temp_factor": 4,  # °C per GC pair
            "melting_temp_offset": 2,  # °C per AT pair
            "codon_length": 3,
            "start_codons": ["ATG"],
            "stop_codons": ["TAA", "TAG", "TGA"],
            "standard_codons": {
                "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
                "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
                "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
                "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
                "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
                "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
                "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
                "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
                "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
                "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
                "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
                "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
                "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
                "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
                "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
                "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G"
            }
        }
    
    def safety_detect(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Check if the DNA query is safe to process."""
        # Check for potentially dangerous sequences
        dangerous_patterns = [
            "pathogen", "virus", "toxin", "biohazard",
            "radioactive", "prion", "carcinogen"
        ]
        
        for key, value in inputs.items():
            if isinstance(value, str):
                for pattern in dangerous_patterns:
                    if pattern in value.lower():
                        return {
                            "safe": False,
                            "reason": f"Potentially dangerous sequence detected: {pattern}",
                            "allowed": False
                        }
        
        return {"safe": True, "reason": "No safety concerns detected", "allowed": True}
    
    def run(
        self,
        sequence: str,
        analysis_type: str = "basic",
        temperature: Optional[float] = None,
        ph: Optional[float] = None,
        salt_concentration: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Analyze DNA sequence and perform various calculations.
        
        Args:
            sequence: DNA sequence to analyze
            analysis_type: Type of analysis ("basic", "translation", "structure", etc.)
            temperature: Temperature in °C for melting calculations
            ph: pH value for stability calculations
            salt_concentration: Salt concentration in mM for stability calculations
            
        Returns:
            Dictionary with analysis results
        """
        # Check safety first
        inputs = {"sequence": sequence}
        safety_result = self.safety_detect(inputs)
        if not safety_result.get("allowed", True):
            return {
                "success": False,
                "error": "Safety check failed",
                "details": safety_result
            }
        
        # Normalize sequence
        sequence = sequence.upper().strip()
        result = {"success": True, "analysis_type": analysis_type}
        
        # Basic sequence analysis
        if analysis_type.lower() in ["basic", "all"]:
            result["length"] = len(sequence)
            result["gc_content"] = f"{(sequence.count('G') + sequence.count('C'))/len(sequence)*100:.1f}%"
            result["at_content"] = f"{(sequence.count('A') + sequence.count('T'))/len(sequence)*100:.1f}%"
            
            # Calculate melting temperature
            gc_count = sequence.count('G') + sequence.count('C')
            at_count = sequence.count('A') + sequence.count('T')
            melting_temp = (gc_count * self.constants["melting_temp_factor"] + 
                          at_count * self.constants["melting_temp_offset"])
            result["melting_temperature"] = f"{melting_temp}°C"
            
            # Calculate physical length
            result["physical_length"] = f"{len(sequence) * self.constants['helix_rise']:.2f} nm"
        
        # Translation analysis
        if analysis_type.lower() in ["translation", "all"]:
            # Find all potential ORFs
            orfs = []
            for frame in range(3):
                for i in range(frame, len(sequence)-2, 3):
                    codon = sequence[i:i+3]
                    if codon in self.constants["start_codons"]:
                        # Look for stop codon
                        for j in range(i+3, len(sequence)-2, 3):
                            if sequence[j:j+3] in self.constants["stop_codons"]:
                                orfs.append({
                                    "start": i,
                                    "end": j+3,
                                    "length": j+3-i,
                                    "frame": frame+1
                                })
                                break
            
            result["open_reading_frames"] = orfs
            
            # Translate sequence
            protein = ""
            for i in range(0, len(sequence)-2, 3):
                codon = sequence[i:i+3]
                if codon in self.constants["standard_codons"]:
                    protein += self.constants["standard_codons"][codon]
                else:
                    protein += "X"  # Unknown codon
            
            result["translated_sequence"] = protein
        
        # Secondary structure prediction
        if analysis_type.lower() in ["structure", "all"]:
            # Simple hairpin detection
            hairpins = []
            for i in range(len(sequence)-4):
                for j in range(i+4, len(sequence)):
                    if self._is_complementary(sequence[i], sequence[j]):
                        hairpins.append({
                            "position": i,
                            "length": j-i+1,
                            "type": "potential_hairpin"
                        })
            
            result["secondary_structures"] = hairpins
        
        # Stability analysis
        if analysis_type.lower() in ["stability", "all"]:
            if temperature is not None:
                result["temperature_stability"] = self._calculate_temperature_stability(
                    sequence, temperature
                )
            
            if ph is not None:
                result["ph_stability"] = self._calculate_ph_stability(sequence, ph)
            
            if salt_concentration is not None:
                result["salt_stability"] = self._calculate_salt_stability(
                    sequence, salt_concentration
                )
        
        return result
    
    def _is_complementary(self, base1: str, base2: str) -> bool:
        """Check if two bases are complementary."""
        pairs = {"A": "T", "T": "A", "G": "C", "C": "G"}
        return pairs.get(base1) == base2
    
    def _calculate_temperature_stability(self, sequence: str, temperature: float) -> str:
        """Calculate temperature stability of the sequence."""
        gc_count = sequence.count('G') + sequence.count('C')
        at_count = sequence.count('A') + sequence.count('T')
        melting_temp = (gc_count * self.constants["melting_temp_factor"] + 
                       at_count * self.constants["melting_temp_offset"])
        
        if abs(temperature - melting_temp) < 5:
            return "unstable (near melting temperature)"
        elif temperature < melting_temp - 10:
            return "stable"
        else:
            return "partially stable"
    
    def _calculate_ph_stability(self, sequence: str, ph: float) -> str:
        """Calculate pH stability of the sequence."""
        if 6.5 <= ph <= 8.0:
            return "stable"
        elif 5.5 <= ph <= 9.0:
            return "partially stable"
        else:
            return "unstable"
    
    def _calculate_salt_stability(self, sequence: str, salt_concentration: float) -> str:
        """Calculate salt concentration stability of the sequence."""
        if 50 <= salt_concentration <= 200:  # mM
            return "stable"
        elif 10 <= salt_concentration <= 500:
            return "partially stable"
        else:
            return "unstable"


class ProteinTool(BaseTool):
    """Tool for protein sequence analysis and structure prediction."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the tool with a language model for protein analysis."""
        self.client, self.model = create_client(model)
        # Common protein properties and constants
        self.properties = {
            "amino_acids": {
                "A": {"mw": 89.09, "hydropathy": 1.8, "pka": None},
                "R": {"mw": 174.20, "hydropathy": -4.5, "pka": 12.48},
                "N": {"mw": 132.12, "hydropathy": -3.5, "pka": None},
                "D": {"mw": 133.10, "hydropathy": -3.5, "pka": 3.65},
                "C": {"mw": 121.15, "hydropathy": 2.5, "pka": 8.18},
                "Q": {"mw": 146.15, "hydropathy": -3.5, "pka": None},
                "E": {"mw": 147.13, "hydropathy": -3.5, "pka": 4.25},
                "G": {"mw": 75.07, "hydropathy": -0.4, "pka": None},
                "H": {"mw": 155.16, "hydropathy": -3.2, "pka": 6.00},
                "I": {"mw": 131.17, "hydropathy": 4.5, "pka": None},
                "L": {"mw": 131.17, "hydropathy": 3.8, "pka": None},
                "K": {"mw": 146.19, "hydropathy": -3.9, "pka": 10.53},
                "M": {"mw": 149.21, "hydropathy": 1.9, "pka": None},
                "F": {"mw": 165.19, "hydropathy": 2.8, "pka": None},
                "P": {"mw": 115.13, "hydropathy": -1.6, "pka": None},
                "S": {"mw": 105.09, "hydropathy": -0.8, "pka": None},
                "T": {"mw": 119.12, "hydropathy": -0.7, "pka": None},
                "W": {"mw": 204.23, "hydropathy": -0.9, "pka": None},
                "Y": {"mw": 181.19, "hydropathy": -1.3, "pka": 10.07},
                "V": {"mw": 117.15, "hydropathy": 4.2, "pka": None}
            },
            "secondary_structure": {
                "helix": ["E", "A", "L", "M", "Q", "K", "R", "H"],
                "sheet": ["V", "I", "Y", "F", "W", "T"],
                "turn": ["P", "G", "D", "N", "S"]
            }
        }
    
    def safety_detect(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Check if the protein query is safe to process."""
        # Check for potentially dangerous proteins
        dangerous_patterns = [
            "toxin", "prion", "venom", "allergen",
            "pathogen", "virus", "biohazard"
        ]
        
        for key, value in inputs.items():
            if isinstance(value, str):
                for pattern in dangerous_patterns:
                    if pattern in value.lower():
                        return {
                            "safe": False,
                            "reason": f"Potentially dangerous protein detected: {pattern}",
                            "allowed": False
                        }
        
        return {"safe": True, "reason": "No safety concerns detected", "allowed": True}
    
    def run(
        self,
        sequence: str,
        analysis_type: str = "basic",
        ph: Optional[float] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Analyze protein sequence and predict properties.
        
        Args:
            sequence: Protein sequence to analyze
            analysis_type: Type of analysis ("basic", "structure", "stability", etc.)
            ph: pH value for stability calculations
            temperature: Temperature in °C for stability calculations
            
        Returns:
            Dictionary with analysis results
        """
        # Check safety first
        inputs = {"sequence": sequence}
        safety_result = self.safety_detect(inputs)
        if not safety_result.get("allowed", True):
            return {
                "success": False,
                "error": "Safety check failed",
                "details": safety_result
            }
        
        # Normalize sequence
        sequence = sequence.upper().strip()
        result = {"success": True, "analysis_type": analysis_type}
        
        # Basic sequence analysis
        if analysis_type.lower() in ["basic", "all"]:
            result["length"] = len(sequence)
            result["molecular_weight"] = self._calculate_molecular_weight(sequence)
            result["amino_acid_composition"] = self._calculate_amino_acid_composition(sequence)
            result["isoelectric_point"] = self._calculate_isoelectric_point(sequence)
            result["extinction_coefficient"] = self._calculate_extinction_coefficient(sequence)
        
        # Secondary structure prediction
        if analysis_type.lower() in ["structure", "all"]:
            result["secondary_structure"] = self._predict_secondary_structure(sequence)
            result["hydrophobicity"] = self._calculate_hydrophobicity(sequence)
        
        # Stability analysis
        if analysis_type.lower() in ["stability", "all"]:
            if ph is not None:
                result["ph_stability"] = self._calculate_ph_stability(sequence, ph)
            if temperature is not None:
                result["temperature_stability"] = self._calculate_temperature_stability(
                    sequence, temperature
                )
        
        return result
    
    def _calculate_molecular_weight(self, sequence: str) -> float:
        """Calculate molecular weight of the protein."""
        mw = 0
        for aa in sequence:
            if aa in self.properties["amino_acids"]:
                mw += self.properties["amino_acids"][aa]["mw"]
        return mw - (len(sequence)-1)*18.02  # Subtract water molecules for peptide bonds
    
    def _calculate_amino_acid_composition(self, sequence: str) -> Dict[str, float]:
        """Calculate amino acid composition percentages."""
        composition = {}
        for aa in self.properties["amino_acids"]:
            count = sequence.count(aa)
            composition[aa] = f"{count/len(sequence)*100:.1f}%"
        return composition
    
    def _calculate_isoelectric_point(self, sequence: str) -> float:
        """Calculate isoelectric point of the protein."""
        # Simplified calculation based on charged residues
        charged_residues = {
            "D": -1, "E": -1,  # Acidic
            "K": 1, "R": 1, "H": 0.1  # Basic
        }
        
        net_charge = 0
        for aa in sequence:
            if aa in charged_residues:
                net_charge += charged_residues[aa]
        
        # Approximate pI calculation
        return 7.0 - net_charge/len(sequence)
    
    def _calculate_extinction_coefficient(self, sequence: str) -> float:
        """Calculate extinction coefficient at 280 nm."""
        # Based on Trp, Tyr, and Cys content
        extinction = (
            sequence.count("W") * 5500 +  # Tryptophan
            sequence.count("Y") * 1490 +  # Tyrosine
            sequence.count("C") * 125     # Cysteine (reduced)
        )
        return extinction
    
    def _predict_secondary_structure(self, sequence: str) -> Dict[str, float]:
        """Predict secondary structure content."""
        helix_count = sum(1 for aa in sequence if aa in self.properties["secondary_structure"]["helix"])
        sheet_count = sum(1 for aa in sequence if aa in self.properties["secondary_structure"]["sheet"])
        turn_count = sum(1 for aa in sequence if aa in self.properties["secondary_structure"]["turn"])
        
        return {
            "helix": f"{helix_count/len(sequence)*100:.1f}%",
            "sheet": f"{sheet_count/len(sequence)*100:.1f}%",
            "turn": f"{turn_count/len(sequence)*100:.1f}%"
        }
    
    def _calculate_hydrophobicity(self, sequence: str) -> float:
        """Calculate average hydrophobicity of the protein."""
        hydropathy = 0
        for aa in sequence:
            if aa in self.properties["amino_acids"]:
                hydropathy += self.properties["amino_acids"][aa]["hydropathy"]
        return hydropathy/len(sequence)
    
    def _calculate_ph_stability(self, sequence: str, ph: float) -> str:
        """Calculate pH stability of the protein."""
        pI = self._calculate_isoelectric_point(sequence)
        if abs(ph - pI) < 1:
            return "unstable (near isoelectric point)"
        elif abs(ph - pI) < 2:
            return "partially stable"
        else:
            return "stable"
    
    def _calculate_temperature_stability(self, sequence: str, temperature: float) -> str:
        """Calculate temperature stability of the protein."""
        # Simplified stability prediction based on composition
        hydrophobic_ratio = sum(1 for aa in sequence if self.properties["amino_acids"][aa]["hydropathy"] > 0)/len(sequence)
        
        if temperature > 60:
            return "unstable" if hydrophobic_ratio < 0.4 else "partially stable"
        elif temperature > 40:
            return "partially stable" if hydrophobic_ratio < 0.3 else "stable"
        else:
            return "stable" 