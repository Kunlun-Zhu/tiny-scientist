import json
import math
from typing import Any, Dict, List, Optional, Union

from ..tool import BaseTool
from ..utils.llm import create_client, get_response_from_llm


class MedicalDiagnosisTool(BaseTool):
    """Tool for medical diagnosis and symptom analysis."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the tool with a language model for medical analysis."""
        self.client, self.model = create_client(model)
        # Common conditions database
        self.conditions = {
            "common_cold": {
                "symptoms": ["runny_nose", "sore_throat", "cough", "sneezing", "mild_fever"],
                "duration": "7-10 days",
                "treatment": ["rest", "fluids", "over_the_counter_meds"],
                "severity": "mild"
            },
            "influenza": {
                "symptoms": ["high_fever", "body_aches", "fatigue", "cough", "headache"],
                "duration": "1-2 weeks",
                "treatment": ["rest", "fluids", "antiviral_meds"],
                "severity": "moderate"
            },
            "hypertension": {
                "symptoms": ["headache", "dizziness", "blurred_vision", "shortness_of_breath"],
                "duration": "chronic",
                "treatment": ["lifestyle_changes", "medication"],
                "severity": "serious"
            },
            "diabetes": {
                "symptoms": ["increased_thirst", "frequent_urination", "fatigue", "blurred_vision"],
                "duration": "chronic",
                "treatment": ["insulin", "diet_control", "exercise"],
                "severity": "serious"
            }
        }
        
        # Vital signs reference ranges
        self.vital_ranges = {
            "temperature": {"normal": (36.5, 37.5), "fever": (37.6, 40.0), "hypothermia": (0, 35.0)},
            "heart_rate": {
                "adult": (60, 100),
                "child": (70, 120),
                "infant": (100, 160)
            },
            "blood_pressure": {
                "normal": (90, 120),
                "elevated": (121, 129),
                "hypertension": (130, 180)
            },
            "respiratory_rate": {
                "adult": (12, 20),
                "child": (20, 30),
                "infant": (30, 60)
            }
        }
    
    def safety_detect(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Check if the medical query is safe to process."""
        # Check for emergency symptoms
        emergency_patterns = [
            "chest_pain", "difficulty_breathing", "severe_bleeding",
            "loss_of_consciousness", "severe_burns", "seizure",
            "stroke_symptoms", "severe_allergic_reaction"
        ]
        
        for key, value in inputs.items():
            if isinstance(value, str):
                for pattern in emergency_patterns:
                    if pattern in value.lower():
                        return {
                            "safe": False,
                            "reason": f"Emergency symptom detected: {pattern}",
                            "allowed": False,
                            "action": "Seek immediate medical attention"
                        }
        
        return {"safe": True, "reason": "No emergency symptoms detected", "allowed": True}
    
    def run(
        self,
        symptoms: List[str],
        age: Optional[int] = None,
        gender: Optional[str] = None,
        vital_signs: Optional[Dict[str, float]] = None,
        medical_history: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze symptoms and provide potential diagnoses.
        
        Args:
            symptoms: List of symptoms
            age: Patient's age
            gender: Patient's gender
            vital_signs: Dictionary of vital signs
            medical_history: List of previous medical conditions
            
        Returns:
            Dictionary with analysis results
        """
        # Check safety first
        inputs = {"symptoms": " ".join(symptoms)}
        safety_result = self.safety_detect(inputs)
        if not safety_result.get("allowed", True):
            return {
                "success": False,
                "error": "Safety check failed",
                "details": safety_result
            }
        
        result = {"success": True}
        
        # Validate age
        if age is not None:
            if age < 0 or age > 120:
                return {
                    "success": False,
                    "error": "Invalid age provided"
                }
            result["age"] = age
        
        # Analyze vital signs
        if vital_signs:
            result["vital_signs_analysis"] = self._analyze_vital_signs(vital_signs, age)
        
        # Match symptoms to conditions
        matched_conditions = []
        for condition, data in self.conditions.items():
            symptom_matches = sum(1 for symptom in symptoms if symptom in data["symptoms"])
            if symptom_matches >= 2:  # Require at least 2 matching symptoms
                matched_conditions.append({
                    "condition": condition,
                    "matching_symptoms": symptom_matches,
                    "total_symptoms": len(data["symptoms"]),
                    "severity": data["severity"],
                    "treatment": data["treatment"]
                })
        
        result["potential_diagnoses"] = sorted(
            matched_conditions,
            key=lambda x: x["matching_symptoms"],
            reverse=True
        )
        
        # Consider medical history
        if medical_history:
            result["medical_history_impact"] = self._analyze_medical_history(
                medical_history,
                matched_conditions
            )
        
        return result
    
    def _analyze_vital_signs(self, vital_signs: Dict[str, float], age: Optional[int]) -> Dict[str, Any]:
        """Analyze vital signs against reference ranges."""
        analysis = {}
        
        # Temperature analysis
        if "temperature" in vital_signs:
            temp = vital_signs["temperature"]
            if temp < self.vital_ranges["temperature"]["normal"][0]:
                analysis["temperature"] = "hypothermia"
            elif temp > self.vital_ranges["temperature"]["normal"][1]:
                analysis["temperature"] = "fever"
            else:
                analysis["temperature"] = "normal"
        
        # Heart rate analysis
        if "heart_rate" in vital_signs:
            hr = vital_signs["heart_rate"]
            if age is not None:
                if age < 1:
                    range_key = "infant"
                elif age < 12:
                    range_key = "child"
                else:
                    range_key = "adult"
                
                normal_range = self.vital_ranges["heart_rate"][range_key]
                if hr < normal_range[0]:
                    analysis["heart_rate"] = "bradycardia"
                elif hr > normal_range[1]:
                    analysis["heart_rate"] = "tachycardia"
                else:
                    analysis["heart_rate"] = "normal"
        
        # Blood pressure analysis
        if "blood_pressure" in vital_signs:
            bp = vital_signs["blood_pressure"]
            if bp < self.vital_ranges["blood_pressure"]["normal"][0]:
                analysis["blood_pressure"] = "hypotension"
            elif bp > self.vital_ranges["blood_pressure"]["hypertension"][0]:
                analysis["blood_pressure"] = "hypertension"
            else:
                analysis["blood_pressure"] = "normal"
        
        return analysis
    
    def _analyze_medical_history(
        self,
        medical_history: List[str],
        matched_conditions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze impact of medical history on potential diagnoses."""
        impact = {}
        
        for condition in matched_conditions:
            condition_name = condition["condition"]
            if condition_name in medical_history:
                impact[condition_name] = {
                    "recurrence_risk": "high",
                    "severity_increase": "possible"
                }
            else:
                impact[condition_name] = {
                    "recurrence_risk": "low",
                    "severity_increase": "unlikely"
                }
        
        return impact


class MedicalAnalysisTool(BaseTool):
    """Tool for medical calculations and analysis."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the tool with a language model for medical analysis."""
        self.client, self.model = create_client(model)
        # Medical constants and formulas
        self.constants = {
            "bmi_ranges": {
                "underweight": (0, 18.5),
                "normal": (18.5, 24.9),
                "overweight": (25, 29.9),
                "obese": (30, 100)
            },
            "gfr_formula": {
                "male": lambda age, creatinine, weight: (140 - age) * weight / (72 * creatinine),
                "female": lambda age, creatinine, weight: 0.85 * (140 - age) * weight / (72 * creatinine)
            },
            "ideal_weight_formula": {
                "male": lambda height: 50 + 2.3 * (height - 60),
                "female": lambda height: 45.5 + 2.3 * (height - 60)
            },
            "calorie_needs": {
                "sedentary": 1.2,
                "light": 1.375,
                "moderate": 1.55,
                "active": 1.725,
                "very_active": 1.9
            }
        }
    
    def safety_detect(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Check if the medical calculation query is safe to process."""
        # Check for potentially dangerous calculations
        dangerous_patterns = [
            "overdose", "toxic", "lethal", "critical",
            "emergency", "severe", "extreme"
        ]
        
        for key, value in inputs.items():
            if isinstance(value, str):
                for pattern in dangerous_patterns:
                    if pattern in value.lower():
                        return {
                            "safe": False,
                            "reason": f"Potentially dangerous calculation detected: {pattern}",
                            "allowed": False
                        }
        
        return {"safe": True, "reason": "No safety concerns detected", "allowed": True}
    
    def run(
        self,
        calculation_type: str,
        parameters: Dict[str, Any],
        units: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Perform medical calculations and analysis.
        
        Args:
            calculation_type: Type of calculation ("bmi", "gfr", "ideal_weight", etc.)
            parameters: Dictionary of calculation parameters
            units: Dictionary of unit specifications
            
        Returns:
            Dictionary with calculation results
        """
        # Check safety first
        safety_result = self.safety_detect({"type": calculation_type})
        if not safety_result.get("allowed", True):
            return {
                "success": False,
                "error": "Safety check failed",
                "details": safety_result
            }
        
        result = {"success": True, "calculation_type": calculation_type}
        
        # BMI calculation
        if calculation_type.lower() == "bmi":
            if "weight" in parameters and "height" in parameters:
                weight = parameters["weight"]
                height = parameters["height"]
                bmi = weight / (height ** 2)
                result["bmi"] = bmi
                result["category"] = self._get_bmi_category(bmi)
        
        # GFR calculation
        elif calculation_type.lower() == "gfr":
            if all(k in parameters for k in ["age", "creatinine", "weight", "gender"]):
                age = parameters["age"]
                creatinine = parameters["creatinine"]
                weight = parameters["weight"]
                gender = parameters["gender"]
                
                formula = self.constants["gfr_formula"][gender.lower()]
                gfr = formula(age, creatinine, weight)
                result["gfr"] = gfr
                result["kidney_function"] = self._get_kidney_function(gfr)
        
        # Ideal weight calculation
        elif calculation_type.lower() == "ideal_weight":
            if "height" in parameters and "gender" in parameters:
                height = parameters["height"]
                gender = parameters["gender"]
                
                formula = self.constants["ideal_weight_formula"][gender.lower()]
                ideal_weight = formula(height)
                result["ideal_weight"] = ideal_weight
        
        # Calorie needs calculation
        elif calculation_type.lower() == "calorie_needs":
            if all(k in parameters for k in ["weight", "height", "age", "gender", "activity_level"]):
                bmr = self._calculate_bmr(
                    parameters["weight"],
                    parameters["height"],
                    parameters["age"],
                    parameters["gender"]
                )
                activity_factor = self.constants["calorie_needs"][parameters["activity_level"].lower()]
                result["daily_calories"] = bmr * activity_factor
        
        return result
    
    def _get_bmi_category(self, bmi: float) -> str:
        """Get BMI category based on calculated BMI."""
        for category, (lower, upper) in self.constants["bmi_ranges"].items():
            if lower <= bmi <= upper:
                return category
        return "invalid"
    
    def _get_kidney_function(self, gfr: float) -> str:
        """Get kidney function category based on GFR."""
        if gfr >= 90:
            return "normal"
        elif gfr >= 60:
            return "mildly_decreased"
        elif gfr >= 30:
            return "moderately_decreased"
        elif gfr >= 15:
            return "severely_decreased"
        else:
            return "kidney_failure"
    
    def _calculate_bmr(self, weight: float, height: float, age: int, gender: str) -> float:
        """Calculate Basal Metabolic Rate using Mifflin-St Jeor Equation."""
        if gender.lower() == "male":
            return 10 * weight + 6.25 * height - 5 * age + 5
        else:
            return 10 * weight + 6.25 * height - 5 * age - 161 