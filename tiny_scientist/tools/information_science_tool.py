import json
import math
import zlib
from collections import Counter
from typing import Any, Dict, List, Optional, Union

from ..tool import BaseTool
from ..utils.llm import create_client, get_response_from_llm


class InformationTheoryTool(BaseTool):
    """Tool for information theory calculations and analysis."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the tool with a language model for information analysis."""
        self.client, self.model = create_client(model)
        # Common information theory constants and properties
        self.constants = {
            "shannon_entropy": True,
            "compression_ratio": True,
            "information_density": True,
            "encoding_schemes": {
                "utf-8": 1,
                "ascii": 1,
                "base64": 4/3,
                "hex": 2,
                "binary": 8
            }
        }
    
    def safety_detect(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Check if the information query is safe to process."""
        # Check for potentially sensitive information
        sensitive_patterns = [
            "password", "credit card", "ssn", "social security",
            "private key", "secret", "confidential"
        ]
        
        for key, value in inputs.items():
            if isinstance(value, str):
                for pattern in sensitive_patterns:
                    if pattern in value.lower():
                        return {
                            "safe": False,
                            "reason": f"Potentially sensitive information detected: {pattern}",
                            "allowed": False
                        }
        
        return {"safe": True, "reason": "No safety concerns detected", "allowed": True}
    
    def run(
        self,
        data: Union[str, List[Any]],
        analysis_type: str = "basic",
        encoding: Optional[str] = None,
        compression: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze data using information theory concepts.
        
        Args:
            data: Data to analyze (string or list)
            analysis_type: Type of analysis ("basic", "entropy", "compression", etc.)
            encoding: Encoding scheme to use
            compression: Compression algorithm to use
            
        Returns:
            Dictionary with analysis results
        """
        # Check safety first
        inputs = {"data": str(data)}
        safety_result = self.safety_detect(inputs)
        if not safety_result.get("allowed", True):
            return {
                "success": False,
                "error": "Safety check failed",
                "details": safety_result
            }
        
        result = {"success": True, "analysis_type": analysis_type}
        
        # Basic information analysis
        if analysis_type.lower() in ["basic", "all"]:
            if isinstance(data, str):
                result["length"] = len(data)
                result["unique_elements"] = len(set(data))
                result["information_density"] = self._calculate_information_density(data)
            
            if isinstance(data, list):
                result["length"] = len(data)
                result["unique_elements"] = len(set(data))
                result["information_density"] = self._calculate_information_density(str(data))
        
        # Entropy analysis
        if analysis_type.lower() in ["entropy", "all"]:
            if isinstance(data, str):
                result["shannon_entropy"] = self._calculate_shannon_entropy(data)
                result["bits_per_character"] = result["shannon_entropy"]
                result["information_content"] = f"{result['shannon_entropy'] * len(data):.2f} bits"
            
            if isinstance(data, list):
                result["shannon_entropy"] = self._calculate_shannon_entropy(str(data))
                result["bits_per_element"] = result["shannon_entropy"]
                result["information_content"] = f"{result['shannon_entropy'] * len(data):.2f} bits"
        
        # Compression analysis
        if analysis_type.lower() in ["compression", "all"]:
            if isinstance(data, str):
                result["compression_ratio"] = self._calculate_compression_ratio(data)
                result["compressed_size"] = self._calculate_compressed_size(data)
            
            if isinstance(data, list):
                result["compression_ratio"] = self._calculate_compression_ratio(str(data))
                result["compressed_size"] = self._calculate_compressed_size(str(data))
        
        # Encoding analysis
        if analysis_type.lower() in ["encoding", "all"] and encoding:
            if isinstance(data, str):
                result["encoded_size"] = self._calculate_encoded_size(data, encoding)
                result["encoding_efficiency"] = self._calculate_encoding_efficiency(data, encoding)
            
            if isinstance(data, list):
                result["encoded_size"] = self._calculate_encoded_size(str(data), encoding)
                result["encoding_efficiency"] = self._calculate_encoding_efficiency(str(data), encoding)
        
        return result
    
    def _calculate_shannon_entropy(self, data: str) -> float:
        """Calculate Shannon entropy of the data."""
        counts = Counter(data)
        probabilities = [count/len(data) for count in counts.values()]
        return -sum(p * math.log2(p) for p in probabilities)
    
    def _calculate_information_density(self, data: str) -> float:
        """Calculate information density of the data."""
        unique_chars = len(set(data))
        total_chars = len(data)
        return unique_chars/total_chars if total_chars > 0 else 0
    
    def _calculate_compression_ratio(self, data: str) -> float:
        """Calculate compression ratio of the data."""
        original_size = len(data.encode('utf-8'))
        compressed_size = len(zlib.compress(data.encode('utf-8')))
        return original_size/compressed_size if compressed_size > 0 else 1
    
    def _calculate_compressed_size(self, data: str) -> int:
        """Calculate compressed size of the data."""
        return len(zlib.compress(data.encode('utf-8')))
    
    def _calculate_encoded_size(self, data: str, encoding: str) -> int:
        """Calculate encoded size of the data."""
        if encoding.lower() in self.constants["encoding_schemes"]:
            multiplier = self.constants["encoding_schemes"][encoding.lower()]
            return int(len(data) * multiplier)
        return len(data)
    
    def _calculate_encoding_efficiency(self, data: str, encoding: str) -> float:
        """Calculate encoding efficiency of the data."""
        if encoding.lower() in self.constants["encoding_schemes"]:
            multiplier = self.constants["encoding_schemes"][encoding.lower()]
            return 1/multiplier
        return 1


class DataAnalysisTool(BaseTool):
    """Tool for data analysis and statistical calculations."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the tool with a language model for data analysis."""
        self.client, self.model = create_client(model)
        # Common statistical constants and properties
        self.constants = {
            "confidence_levels": {
                "90%": 1.645,
                "95%": 1.96,
                "99%": 2.576
            },
            "distribution_types": [
                "normal", "uniform", "binomial", "poisson",
                "exponential", "geometric", "hypergeometric"
            ]
        }
    
    def safety_detect(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Check if the data analysis query is safe to process."""
        # Check for potentially sensitive data
        sensitive_patterns = [
            "personal", "private", "confidential", "sensitive",
            "restricted", "classified", "proprietary"
        ]
        
        for key, value in inputs.items():
            if isinstance(value, str):
                for pattern in sensitive_patterns:
                    if pattern in value.lower():
                        return {
                            "safe": False,
                            "reason": f"Potentially sensitive data detected: {pattern}",
                            "allowed": False
                        }
        
        return {"safe": True, "reason": "No safety concerns detected", "allowed": True}
    
    def run(
        self,
        data: List[float],
        analysis_type: str = "basic",
        confidence_level: Optional[str] = None,
        distribution_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze data using statistical methods.
        
        Args:
            data: List of numerical data points
            analysis_type: Type of analysis ("basic", "descriptive", "inferential", etc.)
            confidence_level: Confidence level for interval estimates
            distribution_type: Assumed distribution type
            
        Returns:
            Dictionary with analysis results
        """
        # Check safety first
        inputs = {"data": str(data)}
        safety_result = self.safety_detect(inputs)
        if not safety_result.get("allowed", True):
            return {
                "success": False,
                "error": "Safety check failed",
                "details": safety_result
            }
        
        result = {"success": True, "analysis_type": analysis_type}
        
        # Basic descriptive statistics
        if analysis_type.lower() in ["basic", "all"]:
            result["count"] = len(data)
            result["mean"] = self._calculate_mean(data)
            result["median"] = self._calculate_median(data)
            result["mode"] = self._calculate_mode(data)
            result["range"] = self._calculate_range(data)
            result["variance"] = self._calculate_variance(data)
            result["standard_deviation"] = self._calculate_standard_deviation(data)
        
        # Distribution analysis
        if analysis_type.lower() in ["distribution", "all"] and distribution_type:
            if distribution_type.lower() in self.constants["distribution_types"]:
                result["distribution_fit"] = self._fit_distribution(data, distribution_type)
                result["goodness_of_fit"] = self._calculate_goodness_of_fit(data, distribution_type)
        
        # Confidence intervals
        if analysis_type.lower() in ["inferential", "all"] and confidence_level:
            if confidence_level in self.constants["confidence_levels"]:
                z_score = self.constants["confidence_levels"][confidence_level]
                result["confidence_interval"] = self._calculate_confidence_interval(
                    data, z_score
                )
        
        return result
    
    def _calculate_mean(self, data: List[float]) -> float:
        """Calculate arithmetic mean of the data."""
        return sum(data)/len(data) if data else 0
    
    def _calculate_median(self, data: List[float]) -> float:
        """Calculate median of the data."""
        sorted_data = sorted(data)
        n = len(sorted_data)
        if n % 2 == 1:
            return sorted_data[n//2]
        return (sorted_data[n//2-1] + sorted_data[n//2])/2
    
    def _calculate_mode(self, data: List[float]) -> List[float]:
        """Calculate mode(s) of the data."""
        counts = Counter(data)
        max_count = max(counts.values())
        return [x for x, count in counts.items() if count == max_count]
    
    def _calculate_range(self, data: List[float]) -> float:
        """Calculate range of the data."""
        return max(data) - min(data) if data else 0
    
    def _calculate_variance(self, data: List[float]) -> float:
        """Calculate variance of the data."""
        mean = self._calculate_mean(data)
        return sum((x - mean)**2 for x in data)/len(data) if data else 0
    
    def _calculate_standard_deviation(self, data: List[float]) -> float:
        """Calculate standard deviation of the data."""
        return math.sqrt(self._calculate_variance(data))
    
    def _fit_distribution(self, data: List[float], distribution_type: str) -> Dict[str, float]:
        """Fit data to specified distribution type."""
        if distribution_type.lower() == "normal":
            return {
                "mean": self._calculate_mean(data),
                "standard_deviation": self._calculate_standard_deviation(data)
            }
        elif distribution_type.lower() == "uniform":
            return {
                "min": min(data),
                "max": max(data)
            }
        return {}
    
    def _calculate_goodness_of_fit(self, data: List[float], distribution_type: str) -> float:
        """Calculate goodness of fit for specified distribution."""
        # Simplified chi-square test
        if distribution_type.lower() == "normal":
            mean = self._calculate_mean(data)
            std = self._calculate_standard_deviation(data)
            expected = [self._normal_pdf(x, mean, std) for x in data]
            observed = [data.count(x)/len(data) for x in set(data)]
            return sum((o-e)**2/e for o, e in zip(observed, expected))
        return 0
    
    def _normal_pdf(self, x: float, mean: float, std: float) -> float:
        """Calculate normal probability density function."""
        return (1/(std * math.sqrt(2*math.pi))) * math.exp(-(x-mean)**2/(2*std**2))
    
    def _calculate_confidence_interval(self, data: List[float], z_score: float) -> Dict[str, float]:
        """Calculate confidence interval for the mean."""
        mean = self._calculate_mean(data)
        std = self._calculate_standard_deviation(data)
        n = len(data)
        margin = z_score * (std/math.sqrt(n))
        return {
            "lower": mean - margin,
            "upper": mean + margin
        } 