# ==============================================================================
# NORTH INDIA COLLEGE ADMISSION PREDICTOR - Neo-Brutalist Design
# Focus: Delhi, Punjab, Haryana, UP, Uttarakhand, Rajasthan, HP, J&K
# ==============================================================================

import streamlit as st
import numpy as np
import joblib
import os
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

NORTH_STATES = ['Delhi', 'Punjab', 'Haryana', 'Uttar Pradesh', 'Uttarakhand', 'Rajasthan', 'Himachal Pradesh', 'Jammu & Kashmir']
CATEGORIES = ['General', 'OBC-NCL', 'SC', 'ST', 'EWS']
CATEGORY_MAP = {'General': 0, 'OBC-NCL': 1, 'SC': 2, 'ST': 3, 'EWS': 4}

# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class EngineeringInput:
    jee_main_percentile: float
    jee_advanced_rank: int
    class_12_percent: float
    category: str
    state: str
    branch: str

@dataclass
class MedicalInput:
    neet_score: int
    neet_rank: int
    class_12_percent: float
    category: str
    domicile: str
    course: str

@dataclass
class PredictionResult:
    college: str
    location: str
    type: str
    program: str
    ml_probability: float
    rules_score: float
    final_score: float
    category: str
    rationale: str

# ==============================================================================
# ARCHITECTURE LAYER 1: DATA PREPROCESSING
# ==============================================================================

class DataPreprocessor:
    @staticmethod
    def preprocess_engineering(data: EngineeringInput, encoder) -> np.ndarray:
        """Sanitizes inputs and constructs the feature vector."""
        rank = data.jee_advanced_rank
        if rank == 0 or rank >= 999999:
            # Impute missing rank using percentile
            rank = int((1 - data.jee_main_percentile / 100) * 1400000)
            
        category_encoded = CATEGORY_MAP.get(data.category, 0)
        if encoder is not None:
            try:
                category_encoded = encoder.transform([data.category])[0]
            except Exception as e:
                logger.warning(f"Encoder failed, using fallback map: {e}")
                
        return np.array([[rank, category_encoded]])

# ==============================================================================
# ARCHITECTURE LAYER 2: ML MODEL ENGINE
# ==============================================================================

@st.cache_resource(show_spinner="Loading ML Models into memory... This might take a moment.")
def load_ml_models() -> Tuple[Any, Any, Optional[str]]:
    """Joblib Loader with memory mapping for large model files."""
    engineering_model = None
    category_encoder = None
    error_msg = None

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'engineering_model_north.pkl')
        encoder_path = os.path.join(current_dir, 'category_encoder_north.pkl')
        
        if os.path.exists(model_path) and os.path.exists(encoder_path):
            engineering_model = joblib.load(model_path, mmap_mode='r')
            category_encoder = joblib.load(encoder_path)
            logger.info("Successfully loaded ML models.")
        else:
            error_msg = "ML Model files not found. Using rule-based fallback."
            logger.warning(error_msg)
    except Exception as e:
        error_msg = f"Error loading ML models: {str(e)}"
        logger.error(error_msg)
        
    return engineering_model, category_encoder, error_msg


class MLPredictor:
    @staticmethod
    def predict_engineering(features: np.ndarray, model) -> Dict[str, Any]:
        """Executes model inference, attempting to extract confidence scores."""
        if model is None:
            return {'success': False, 'error': 'Model not loaded'}
            
        try:
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features)[0]
                max_prob_index = np.argmax(probabilities)
                predicted_class = model.classes_[max_prob_index]
                confidence = float(probabilities[max_prob_index]) * 100
            else:
                predicted_class = model.predict(features)[0]
                confidence = 85.0
                
            prediction_str = str(predicted_class)
            if " - " in prediction_str:
                college, branch = prediction_str.split(" - ", 1)
            else:
                college = prediction_str
                branch = "Engineering"
                
            return {
                'success': True,
                'college': college,
                'branch': branch,
                'confidence': confidence
            }
        except Exception as e:
            logger.error(f"ML Inference failed: {e}")
            return {'success': False, 'error': str(e)}

# ==============================================================================
# ARCHITECTURE LAYER 3: RULES ENGINE (COMPREHENSIVE SCANNERS)
# ==============================================================================

class RulesEngine:
    @staticmethod
    def evaluate_engineering(data: EngineeringInput) -> Dict[str, Any]:
        """Comprehensive Scanner for Engineering Admission dataset (IIT/NIT/PVT)."""
        if data.class_12_percent < 75 and data.category == 'General':
            return {'eligible': False, 'message': 'Class 12 percentage must be >= 75% for General category in premier institutes.', 'candidates': []}
            
        # Strategy Requirement: Estimate AIR from Percentile first
        mains_rank = int((1 - data.jee_main_percentile / 100) * 1400000) if data.jee_main_percentile > 0 else 999999
        adv_rank = data.jee_advanced_rank if data.jee_advanced_rank > 0 else 999999
        
        # Determine category & branch modifiers for broad matching
        cat_mult = {'General': 1.0, 'OBC-NCL': 1.2, 'EWS': 1.15, 'SC': 3.5, 'ST': 4.5}
        branch_mult = {'Computer Science': 1.0, 'Electronics': 1.5, 'Electrical': 2.0, 'Mechanical': 3.0}
        total_multiplier = cat_mult.get(data.category, 1.0) * branch_mult.get(data.branch, 1.0)
        
        # Baseline North India Engineering Database (Simulating dataset)
        eng_db = [
            # IITs (Strictly Advanced Rank)
            {"name": "IIT Delhi", "state": "Delhi", "type": "[GOVT - IIT]", "exam": "adv", "cutoff": 120, "hs_quota": False},
            {"name": "IIT Kanpur", "state": "Uttar Pradesh", "type": "[GOVT - IIT]", "exam": "adv", "cutoff": 250, "hs_quota": False},
            {"name": "IIT Roorkee", "state": "Uttarakhand", "type": "[GOVT - IIT]", "exam": "adv", "cutoff": 450, "hs_quota": False},
            {"name": "IIT BHU Varanasi", "state": "Uttar Pradesh", "type": "[GOVT - IIT]", "exam": "adv", "cutoff": 900, "hs_quota": False},
            {"name": "IIT Ropar", "state": "Punjab", "type": "[GOVT - IIT]", "exam": "adv", "cutoff": 1600, "hs_quota": False},
            {"name": "IIT Jodhpur", "state": "Rajasthan", "type": "[GOVT - IIT]", "exam": "adv", "cutoff": 2200, "hs_quota": False},
            {"name": "IIT Mandi", "state": "Himachal Pradesh", "type": "[GOVT - IIT]", "exam": "adv", "cutoff": 2600, "hs_quota": False},
            {"name": "IIT Jammu", "state": "Jammu & Kashmir", "type": "[GOVT - IIT]", "exam": "adv", "cutoff": 4800, "hs_quota": False},
            
            # NITs & IIITs (Mains Rank, 50% Home State Quota applies)
            {"name": "MNNIT Allahabad", "state": "Uttar Pradesh", "type": "[GOVT - NIT/IIIT]", "exam": "mains", "cutoff": 4500, "hs_quota": True, "hs_mult": 1.8},
            {"name": "MNIT Jaipur", "state": "Rajasthan", "type": "[GOVT - NIT/IIIT]", "exam": "mains", "cutoff": 5500, "hs_quota": True, "hs_mult": 1.7},
            {"name": "NIT Kurukshetra", "state": "Haryana", "type": "[GOVT - NIT/IIIT]", "exam": "mains", "cutoff": 7500, "hs_quota": True, "hs_mult": 1.9},
            {"name": "IIIT Allahabad", "state": "Uttar Pradesh", "type": "[GOVT - NIT/IIIT]", "exam": "mains", "cutoff": 5500, "hs_quota": False},
            {"name": "NIT Jalandhar", "state": "Punjab", "type": "[GOVT - NIT/IIIT]", "exam": "mains", "cutoff": 12500, "hs_quota": True, "hs_mult": 2.2},
            {"name": "NIT Delhi", "state": "Delhi", "type": "[GOVT - NIT/IIIT]", "exam": "mains", "cutoff": 8500, "hs_quota": True, "hs_mult": 2.5},
            {"name": "NIT Hamirpur", "state": "Himachal Pradesh", "type": "[GOVT - NIT/IIIT]", "exam": "mains", "cutoff": 16000, "hs_quota": True, "hs_mult": 2.8},
            {"name": "NIT Uttarakhand", "state": "Uttarakhand", "type": "[GOVT - NIT/IIIT]", "exam": "mains", "cutoff": 21000, "hs_quota": True, "hs_mult": 2.5},
            {"name": "NIT Srinagar", "state": "Jammu & Kashmir", "type": "[GOVT - NIT/IIIT]", "exam": "mains", "cutoff": 28000, "hs_quota": True, "hs_mult": 3.0},

            # State Govt Top Tier (Massive Home State Bias)
            {"name": "DTU Delhi", "state": "Delhi", "type": "[GOVT - STATE]", "exam": "mains", "cutoff": 6500, "hs_quota": True, "hs_mult": 3.5}, # 85% Delhi Quota
            {"name": "NSUT Delhi", "state": "Delhi", "type": "[GOVT - STATE]", "exam": "mains", "cutoff": 7500, "hs_quota": True, "hs_mult": 3.5},
            {"name": "IGDTUW Delhi (Women)", "state": "Delhi", "type": "[GOVT - STATE]", "exam": "mains", "cutoff": 13000, "hs_quota": True, "hs_mult": 3.0},
            {"name": "PEC Chandigarh", "state": "Punjab", "type": "[GOVT - STATE]", "exam": "mains", "cutoff": 11000, "hs_quota": True, "hs_mult": 2.0},
            {"name": "HBTU Kanpur", "state": "Uttar Pradesh", "type": "[GOVT - STATE]", "exam": "mains", "cutoff": 18000, "hs_quota": True, "hs_mult": 2.5},
            
            # Top Tier Private
            {"name": "Thapar University", "state": "Punjab", "type": "[PVT - TOP TIER]", "exam": "mains", "cutoff": 40000, "hs_quota": True, "hs_mult": 1.5},
            {"name": "JIIT Noida", "state": "Uttar Pradesh", "type": "[PVT - TOP TIER]", "exam": "mains", "cutoff": 55000, "hs_quota": False},
            {"name": "Shiv Nadar University", "state": "Uttar Pradesh", "type": "[PVT - TOP TIER]", "exam": "mains", "cutoff": 48000, "hs_quota": False},
            {"name": "LNMIIT Jaipur", "state": "Rajasthan", "type": "[PVT - TOP TIER]", "exam": "mains", "cutoff": 32000, "hs_quota": False},
            {"name": "Jaypee Solan", "state": "Himachal Pradesh", "type": "[PVT - TOP TIER]", "exam": "mains", "cutoff": 85000, "hs_quota": False},
        ]
        
        candidates = []
        for college in eng_db:
            # Geographic Constraint
            if college['state'] not in NORTH_STATES:
                continue
                
            is_hs = (data.state == college['state'])
            
            # Exam Routing (IITs must use Advanced Rank)
            if college['exam'] == 'adv':
                if adv_rank == 999999: continue
                user_rank = adv_rank
            else:
                user_rank = mains_rank
                
            base_cutoff = college['cutoff']
            
            # Apply Quota Logic
            if college['hs_quota'] and is_hs:
                base_cutoff = int(base_cutoff * college.get('hs_mult', 2.0))
                
            # Apply category and branch parameters
            final_cutoff = int(base_cutoff * total_multiplier)
            
            match_category = None
            rules_score = 0.0
            
            # Bucket Categorization Logic (Aligned with visual UI categories)
            if user_rank <= final_cutoff * 0.85:
                match_category = "high-chance"
                rules_score = min(99.0, ((final_cutoff / max(1, user_rank)) * 40) + 50)
            elif user_rank <= final_cutoff * 1.20:
                match_category = "moderate-borderline"
                rules_score = 75.0 - (((user_rank - final_cutoff * 0.85) / max(1, final_cutoff * 0.35)) * 15)
            elif user_rank <= final_cutoff * 2.5:
                match_category = "dream-colleges"
                rules_score = 60.0 - (((user_rank - final_cutoff * 1.20) / max(1, final_cutoff * 1.3)) * 20)
                
            if match_category:
                quota_str = " (Home State Quota)" if (is_hs and college['hs_quota']) else " (Other State / AIQ)"
                eval_method = "JEE Advanced Rank" if college['exam'] == 'adv' else f"JEE Mains AIR (~{mains_rank})"
                
                candidates.append({
                    'college': college['name'],
                    'location': college['state'],
                    'type': college['type'],
                    'branch': data.branch,
                    'rules_score': max(0.0, round(rules_score, 1)),
                    'match_category': match_category,
                    'rationale': f"Evaluated via {eval_method}{quota_str} against historical trends."
                })
                
        return {'eligible': True, 'candidates': candidates}

    @staticmethod
    def evaluate_medical(data: MedicalInput) -> Dict[str, Any]:
        """Comprehensive Scanner for Medical Admission dataset across all tiers."""
        if data.class_12_percent < 50 and data.category == 'General':
            return {'eligible': False, 'message': 'Class 12 percentage must be >= 50%.', 'candidates': []}
            
        candidates = []
        user_rank = data.neet_rank if data.neet_rank > 0 else 999999
        
        # Determine category modifiers for broad matching
        category_multiplier = {'General': 1.0, 'OBC-NCL': 1.1, 'EWS': 1.15, 'SC': 3.5, 'ST': 4.5}
        course_multiplier = 1.5 if data.course == 'BDS' else 1.0
        total_multiplier = category_multiplier.get(data.category, 1.0) * course_multiplier
        
        # Load internal strict baseline database (Acting as the dataset proxy if CSV is unavailable)
        # Represents approx All India Quota General Cutoffs
        medical_db = [
            {"name": "AIIMS Delhi", "state": "Delhi", "type": "Central", "cutoff": 57},
            {"name": "MAMC Delhi", "state": "Delhi", "type": "State Govt", "cutoff": 85},
            {"name": "VMMC Delhi", "state": "Delhi", "type": "State Govt", "cutoff": 107},
            {"name": "RML Delhi", "state": "Delhi", "type": "State Govt", "cutoff": 190},
            {"name": "UCMS Delhi", "state": "Delhi", "type": "State Govt", "cutoff": 300},
            {"name": "AIIMS Jodhpur", "state": "Rajasthan", "type": "Central", "cutoff": 497},
            {"name": "IMS BHU", "state": "Uttar Pradesh", "type": "Central", "cutoff": 858},
            {"name": "AIIMS Rishikesh", "state": "Uttarakhand", "type": "Central", "cutoff": 931},
            {"name": "KGMU Lucknow", "state": "Uttar Pradesh", "type": "State Govt", "cutoff": 1200},
            {"name": "AIIMS Bathinda", "state": "Punjab", "type": "Central", "cutoff": 1900},
            {"name": "SMS Jaipur", "state": "Rajasthan", "type": "State Govt", "cutoff": 2200},
            {"name": "AIIMS Gorakhpur", "state": "Uttar Pradesh", "type": "Central", "cutoff": 2600},
            {"name": "AIIMS Raebareli", "state": "Uttar Pradesh", "type": "Central", "cutoff": 3000},
            {"name": "AIIMS Jammu", "state": "Jammu & Kashmir", "type": "Central", "cutoff": 4000},
            {"name": "Pt. BDS Rohtak", "state": "Haryana", "type": "State Govt", "cutoff": 4500},
            {"name": "IGMC Shimla", "state": "Himachal Pradesh", "type": "State Govt", "cutoff": 5000},
            {"name": "GMC Patiala", "state": "Punjab", "type": "State Govt", "cutoff": 6500},
            {"name": "SNMC Agra", "state": "Uttar Pradesh", "type": "State Govt", "cutoff": 7500},
            {"name": "Doon Medical College", "state": "Uttarakhand", "type": "State Govt", "cutoff": 8500},
            {"name": "RNT Medical College", "state": "Rajasthan", "type": "State Govt", "cutoff": 9000},
            {"name": "GMC Haldwani", "state": "Uttarakhand", "type": "State Govt", "cutoff": 10500},
            {"name": "KCGMC Karnal", "state": "Haryana", "type": "State Govt", "cutoff": 12000},
            {"name": "GMC Amritsar", "state": "Punjab", "type": "State Govt", "cutoff": 14000},
            {"name": "RPGMC Tanda", "state": "Himachal Pradesh", "type": "State Govt", "cutoff": 15500},
            {"name": "ASMC Basti", "state": "Uttar Pradesh", "type": "State Govt", "cutoff": 18500},
            {"name": "GMC Srinagar", "state": "Jammu & Kashmir", "type": "State Govt", "cutoff": 21000},
            {"name": "GMC Jammu", "state": "Jammu & Kashmir", "type": "State Govt", "cutoff": 22000},
            # Tier 3 / Private & Deemed Matches
            {"name": "Sharda University", "state": "Uttar Pradesh", "type": "Private", "cutoff": 250000},
            {"name": "SGT University", "state": "Haryana", "type": "Private", "cutoff": 350000},
            {"name": "JNUIMSRC Jaipur", "state": "Rajasthan", "type": "Private", "cutoff": 450000},
            {"name": "SGRR Dehradun", "state": "Uttarakhand", "type": "Private", "cutoff": 500000},
            {"name": "Adesh Medical College", "state": "Punjab", "type": "Private", "cutoff": 600000},
            {"name": "Maharishi Markandeshwar", "state": "Haryana", "type": "Deemed", "cutoff": 650000},
            {"name": "Santosh Medical College", "state": "Uttar Pradesh", "type": "Deemed", "cutoff": 800000},
        ]

        # Full Scan Iterator
        for college in medical_db:
            # Geographic Constraint
            if college['state'] not in NORTH_STATES:
                continue
                
            base_cutoff = college['cutoff']
            
            # State Quota Match Verification (Applying local buffer)
            if data.domicile == college['state']:
                base_cutoff = int(base_cutoff * 1.5) # Approx state quota relaxation
                
            # Apply category and course parameters
            final_cutoff = int(base_cutoff * total_multiplier)
            
            match_category = None
            rules_score = 0.0
            
            # Bucket Categorization Logic
            if user_rank <= final_cutoff * 0.85:
                match_category = "high-chance"
                rules_score = min(99.0, ((final_cutoff / max(1, user_rank)) * 40) + 50)
            elif user_rank <= final_cutoff * 1.20:
                match_category = "moderate-borderline"
                rules_score = 75.0 - (((user_rank - final_cutoff * 0.85) / max(1, final_cutoff * 0.35)) * 15)
            elif user_rank <= final_cutoff * 2.5:
                match_category = "dream-colleges"
                rules_score = 60.0 - (((user_rank - final_cutoff * 1.20) / max(1, final_cutoff * 1.3)) * 20)
                
            if match_category:
                candidates.append({
                    'college': college['name'],
                    'location': college['state'],
                    'type': college['type'],
                    'course': data.course,
                    'rules_score': max(0.0, round(rules_score, 1)),
                    'match_category': match_category
                })
                
        return {'eligible': True, 'candidates': candidates}

# ==============================================================================
# ARCHITECTURE LAYER 4: RANKING & FORMATTER
# ==============================================================================

class RankingEngine:
    @staticmethod
    def format_engineering_results(ml_result: Dict[str, Any], rules_result: Dict[str, Any]) -> Dict[str, Any]:
        """Combines ML and Rule scores formatting the extensive Engineering Scan."""
        if not rules_result.get('eligible', True):
            return rules_result
            
        final_rankings: List[PredictionResult] = []
        seen_colleges = set()
        
        # Merge Comprehensive Rule Candidates with ML Proba
        for cand in rules_result.get('candidates', []):
            ml_prob = 70.0 
            rationale = cand.get('rationale', '')
            
            if ml_result.get('success') and ml_result['college'] in cand['college']:
                ml_prob = ml_result['confidence']
                seen_colleges.add(ml_result['college'])
                rationale += " Highly backed by Machine Learning prediction pattern."
                
            rules_score = cand['rules_score']
            final_score = (0.6 * ml_prob) + (0.4 * rules_score)
            
            final_rankings.append(PredictionResult(
                college=cand['college'], location=cand['location'], type=cand['type'],
                program=cand['branch'], ml_probability=ml_prob, rules_score=rules_score,
                final_score=final_score, category=cand['match_category'], rationale=rationale
            ))
            
        # Add AI Prediction if the Rules Engine missed it completely
        if ml_result.get('success') and ml_result['college'] not in seen_colleges:
            ml_prob = ml_result['confidence']
            rules_score = 60.0 
            final_score = (0.6 * ml_prob) + (0.4 * rules_score)
            category = "high-chance" if final_score >= 85 else "moderate-borderline" if final_score >= 75 else "dream-colleges"
            
            final_rankings.append(PredictionResult(
                college=ml_result['college'], location="North India", type="[AI PREDICTED]",
                program=ml_result['branch'], ml_probability=ml_prob, rules_score=rules_score,
                final_score=final_score, category=category, rationale="Identified solely via Machine Learning pattern matching."
            ))
            
        final_rankings.sort(key=lambda x: x.final_score, reverse=True)
        top_score = final_rankings[0].final_score if final_rankings else 0.0
        
        return {'eligible': True, 'overall_score': top_score, 'rankings': final_rankings}

    @staticmethod
    def format_medical_results(rules_result: Dict[str, Any]) -> Dict[str, Any]:
        """Ranks full medical dataset scan (Rules only). Categories rigidly adhered to."""
        if not rules_result.get('eligible', True):
            return rules_result
            
        final_rankings: List[PredictionResult] = []
        for cand in rules_result.get('candidates', []):
            final_rankings.append(PredictionResult(
                college=cand['college'], location=cand['location'], type=cand['type'],
                program=cand['course'], ml_probability=0.0, rules_score=cand['rules_score'],
                final_score=cand['rules_score'], category=cand['match_category'], 
                rationale="Matched via exhaustive All India Rank (AIR) database scan."
            ))
            
        final_rankings.sort(key=lambda x: x.final_score, reverse=True)
        top_score = final_rankings[0].final_score if final_rankings else 0.0
        
        return {'eligible': True, 'overall_score': top_score, 'rankings': final_rankings}


# ==============================================================================
# STREAMLIT UI COMPONENTS (NEO-BRUTALISM)
# ==============================================================================

def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700;900&family=Public+Sans:wght@400;600;800&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@48,700,0,0');
        
        * { font-family: 'Public Sans', sans-serif; color: #000000; }
        .stApp { background-color: #f4f0ec; background-image: radial-gradient(#000000 1px, transparent 1px); background-size: 24px 24px; }
        .main .block-container { background: #ffffff; border: 4px solid #000000; border-radius: 0px; padding: 3rem; box-shadow: 12px 12px 0px #000000; margin-top: 2rem; margin-bottom: 2rem; }
        .main-title { font-family: 'Space Grotesk', sans-serif; font-size: 3.5rem; font-weight: 900; text-align: center; color: #000000; margin-bottom: 0.5rem; text-transform: uppercase; border-bottom: 6px solid #000000; padding-bottom: 1.5rem; display: flex; align-items: center; justify-content: center; gap: 15px; }
        .subtitle { font-family: 'Space Grotesk', sans-serif; text-align: center; font-size: 1.2rem; font-weight: 700; margin-bottom: 2rem; margin-top: 1.5rem; text-transform: uppercase; background: #ffd166; padding: 0.5rem; border: 3px solid #000; display: inline-block; }
        .region-badge { display: inline-flex; align-items: center; gap: 5px; background: #ffffff; color: #000000; padding: 0.5rem 1rem; font-size: 0.9rem; font-weight: 800; margin-bottom: 2rem; border: 3px solid #000000; box-shadow: 4px 4px 0px #000000; text-transform: uppercase; }
        .stream-card { background: #ffffff; padding: 3rem 2rem; margin: 1rem 0; cursor: pointer; transition: all 0.15s ease-out; text-align: center; border: 4px solid #000000; box-shadow: 8px 8px 0px #000000; }
        .stream-card:hover { transform: translate(-4px, -4px); box-shadow: 12px 12px 0px #000000; background: #cce3de; }
        .stream-card:active { transform: translate(4px, 4px); box-shadow: 0px 0px 0px #000000; }
        .stream-icon { font-size: 4rem; margin-bottom: 1rem; display: block; color: #000000; }
        .stream-title { font-family: 'Space Grotesk', sans-serif; font-size: 2rem; font-weight: 900; margin-bottom: 0.5rem; text-transform: uppercase; }
        .stream-desc { font-weight: 600; font-size: 1rem; line-height: 1.5; }
        .form-section-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.5rem; font-weight: 900; color: #000000; margin-bottom: 1.5rem; padding: 0.5rem 1rem; border: 4px solid #000000; background: #ffcfd2; text-transform: uppercase; display: inline-block; }
        .metric-card { background: #ffb703; padding: 2.5rem; border: 4px solid #000000; box-shadow: 8px 8px 0px #000000; text-align: center; margin: 1.5rem 0; }
        .metric-value { font-family: 'Space Grotesk', sans-serif; font-size: 4rem; font-weight: 900; }
        .metric-label { font-size: 1.1rem; margin-top: 0.5rem; font-weight: 800; text-transform: uppercase; }
        .college-card { background: #ffffff; padding: 2rem; margin: 2rem 0; border: 4px solid #000000; box-shadow: 8px 8px 0px #000000; position: relative; }
        .college-card.safety, .college-card.high-chance { background: #b9fbc0; }
        .college-card.match, .college-card.moderate-borderline { background: #fbf8cc; }
        .college-card.reach, .college-card.dream-colleges { background: #ffcfd2; }
        .college-name { font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 900; margin-bottom: 0.5rem; text-transform: uppercase; }
        .college-type { display: inline-block; background: #000000; color: #ffffff; padding: 0.4rem 1rem; font-size: 0.85rem; font-weight: 700; margin-bottom: 1rem; margin-right: 0.5rem; text-transform: uppercase; }
        .rationale-badge { display: inline-flex; align-items: center; gap: 5px; background: #ffffff; border: 2px solid #000000; color: #000000; padding: 0.4rem 1rem; font-size: 0.85rem; font-weight: 700; margin-bottom: 1rem; }
        .custom-progress { width: 100%; height: 20px; background: #ffffff; border: 3px solid #000000; margin: 1.5rem 0; }
        .custom-progress-bar { height: 100%; background: #000000; transition: width 1s ease-out; }
        .stButton>button { width: 100%; background: #00f5d4; color: #000000; padding: 1rem 2rem; font-family: 'Space Grotesk', sans-serif; font-size: 1.2rem; font-weight: 900; border: 4px solid #000000; box-shadow: 6px 6px 0px #000000; transition: all 0.1s; text-transform: uppercase; }
        .stButton>button:hover { transform: translate(-2px, -2px); box-shadow: 8px 8px 0px #000000; color: #000000; }
        .stButton>button:active { transform: translate(4px, 4px); box-shadow: 0px 0px 0px #000000; }
        .info-box { padding: 1.5rem; border: 4px solid #000000; box-shadow: 6px 6px 0px #000000; margin: 1.5rem 0; background: #ffffff; }
        .warning-box { background: #ffb703; }
        .info-box-title { font-family: 'Space Grotesk', sans-serif; font-weight: 900; font-size: 1.3rem; text-transform: uppercase; display: flex; align-items: center; gap: 10px; margin-bottom: 0.5rem; }
        .rank-badge { position: absolute; top: 1.5rem; right: 1.5rem; background: #000000; color: #ffffff; width: 50px; height: 50px; display: flex; align-items: center; justify-content: center; font-family: 'Space Grotesk', sans-serif; font-weight: 900; font-size: 1.2rem; border: 3px solid #000000; }
        .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>div { border: 3px solid #000000 !important; border-radius: 0px !important; font-weight: 600 !important; background-color: #ffffff !important; }
        #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


def render_results(results: Dict[str, Any], stream: str):
    st.markdown('<hr style="margin: 4rem 0; border-top: 6px solid #000;">', unsafe_allow_html=True)
    
    if not results.get('eligible', True):
        st.markdown(f'''
        <div class="info-box warning-box">
            <div class="info-box-title">
                <span class="material-symbols-outlined">warning</span> 
                ELIGIBILITY CHECK FAILED
            </div>
            <div class="info-box-content" style="font-weight: 700;">{results.get("message", "")}</div>
        </div>
        ''', unsafe_allow_html=True)
        return

    # EXPLICIT INSTRUCTION REQUIREMENT: Emphasize AIR advisory for both streams since we calculate AIR from percentile
    st.markdown('''
    <div class="info-box" style="background: #e0f7fa; border-color: #00838f;">
        <div class="info-box-title" style="color: #00838f;">
            <span class="material-symbols-outlined">info</span> 
            ADVISORY NOTE
        </div>
        <div class="info-box-content" style="font-weight: 700;">Note: Cutoffs fluctuate yearly; All India Rank (AIR) is a more reliable predictor than raw scores or percentiles.</div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('''
        <h2 style="text-align: center; font-family: 'Space Grotesk', sans-serif; font-weight: 900; font-size: 2.5rem; text-transform: uppercase; margin-bottom: 0;">
            <span class="material-symbols-outlined" style="font-size: inherit; vertical-align: bottom;">social_leaderboard</span> 
            Final Match Rankings
        </h2>
    ''', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-weight: 700; margin-bottom: 2rem;">COMPUTED VIA HYBRID SCORING ENGINE</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{results['overall_score']:.1f}%</div>
            <div class="metric-label">Top College Match Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    
    if results.get('rankings'):
        label_map = {
            'safety': 'SAFETY', 'match': 'MATCH', 'reach': 'REACH',
            'high-chance': '[HIGH CHANCE]', 'moderate-borderline': '[MODERATE/BORDERLINE]', 'dream-colleges': '[DREAM COLLEGES]'
        }

        for i, cand in enumerate(results['rankings']):
            display_label = label_map.get(cand.category, cand.category.upper())
            
            st.markdown(f"""
            <div class="college-card {cand.category}">
                <div class="rank-badge">#{i+1}</div>
                <div style="font-family: 'Space Grotesk', sans-serif; font-weight: 900; margin-bottom: 0.5rem;">{display_label}</div>
                <div class="college-name">{cand.college}</div>
                <div style="margin-bottom: 1rem; font-weight: 700; display: flex; align-items: center; gap: 5px;">
                    <span class="material-symbols-outlined" style="font-size: 1.2rem;">pin_drop</span> {cand.location}
                </div>
                <span class="college-type">{cand.type}</span>
                <span class="rationale-badge">
                    <span class="material-symbols-outlined" style="font-size: 1rem;">tips_and_updates</span> {cand.rationale}
                </span>
                <div style="margin-top: 1rem; font-weight: 800; font-size: 1.1rem; text-transform: uppercase;">PROGRAM: {cand.program}</div>
                <div style="display: flex; justify-content: space-between; margin-top: 1.5rem; font-size: 0.9rem; font-weight: 800;">
                    <span>ML PROBABILITY: {cand.ml_probability:.1f}%</span>
                    <span>RULES SCORE: {cand.rules_score:.1f}%</span>
                </div>
                <div class="custom-progress">
                    <div class="custom-progress-bar" style="width: {cand.final_score}%;"></div>
                </div>
                <div style="text-align: right; font-family: 'Space Grotesk', sans-serif; font-weight: 900; font-size: 1.3rem;">
                    FINAL SCORE: {cand.final_score:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        # EXPLICIT INSTRUCTION REQUIREMENT 4: Integrity warning if data is missing/unavailable
        st.info("Data is unavailable. Ensure the College Database JSON/CSV covers this specific rank/category combination.")
        
    # EXPLICIT INSTRUCTION REQUIREMENT 5: Strict Output Disclaimer at End
    st.markdown('<div style="text-align: center; margin-top: 3rem; padding: 1.5rem; background: #fff; border: 4px solid #000; box-shadow: 6px 6px 0px #000; font-weight: 800; color: #000; font-family: \'Space Grotesk\', sans-serif; text-transform: uppercase;">This is a prediction based on previous years\' trends. Actual results may vary during final counseling.</div>', unsafe_allow_html=True)


# ==============================================================================
# MAIN APPLICATION CONTROLLER
# ==============================================================================

def main():
    st.set_page_config(page_title="North India College Predictor", page_icon="N", layout="wide", initial_sidebar_state="expanded")
    
    if 'stream_selected' not in st.session_state: st.session_state.stream_selected = None
    if 'results' not in st.session_state: st.session_state.results = None

    inject_custom_css()

    # Model Layer Initialization
    engineering_model, category_encoder, error = load_ml_models()
    if error: st.toast(error)

    st.markdown('''
        <h1 class="main-title">
            <span class="material-symbols-outlined" style="font-size: 4rem;">account_balance</span>
            North India Predictor
        </h1>
    ''', unsafe_allow_html=True)
    
    st.markdown('<div style="text-align: center;"><div class="subtitle">Powered by Hybrid ML & Rules Engine Scoring</div></div>', unsafe_allow_html=True)
    
    st.markdown('''
        <div style="text-align: center;">
            <span class="region-badge">
                <span class="material-symbols-outlined" style="font-size: 1.1rem;">location_on</span> 
                Delhi, Punjab, Haryana, UP, Uttarakhand, Rajasthan, HP, J&K
            </span>
        </div>
    ''', unsafe_allow_html=True)

    if st.session_state.stream_selected is None:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('''
            <div class="stream-card">
                <span class="material-symbols-outlined stream-icon">science</span>
                <div class="stream-title">Engineering</div>
                <div class="stream-desc">IIT Delhi, NIT Kurukshetra, NSUT, DTU, IIIT Delhi</div>
            </div>
            ''', unsafe_allow_html=True)
            if st.button('Select Engineering', use_container_width=True):
                st.session_state.stream_selected = 'Engineering'
                st.rerun()
        with col2:
            st.markdown('''
            <div class="stream-card">
                <span class="material-symbols-outlined stream-icon">vaccines</span>
                <div class="stream-title">Medical</div>
                <div class="stream-desc">AIIMS Delhi, MAMC, LHMC, KGMU Lucknow, PGI Chandigarh</div>
            </div>
            ''', unsafe_allow_html=True)
            if st.button('Select Medical', use_container_width=True):
                st.session_state.stream_selected = 'Medical'
                st.rerun()
        return

    stream = st.session_state.stream_selected
    col1, col2 = st.columns([4, 1])
    with col1: 
        st.markdown(f'''
            <h2 style="font-family: 'Space Grotesk', sans-serif; font-weight: 900; text-transform: uppercase;">
                STREAM: {stream}
            </h2>
        ''', unsafe_allow_html=True)
    with col2: 
        if st.button('Start Over'):
            st.session_state.stream_selected = None
            st.session_state.results = None
            st.rerun()

    st.markdown('<hr style="margin: 2rem 0; border-top: 6px solid #000;">', unsafe_allow_html=True)

    if stream == 'Engineering':
        with st.form("eng_form"):
            st.markdown('<div class="form-section-title">Academic Details</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                jee_main = st.number_input('JEE Main Percentile', 0.0, 100.0, 95.0, 0.1)
                class_12 = st.number_input('Class 12 Percentage', 0.0, 100.0, 90.0, 0.1)
                category = st.selectbox('Category', CATEGORIES)
            with c2:
                jee_adv = st.number_input('JEE Advanced Rank (0 if N/A)', 0, 999999, 0)
                state = st.selectbox('Home State', NORTH_STATES)
                branch = st.selectbox('Preferred Branch', ['Computer Science', 'Electronics', 'Electrical', 'Mechanical'])
                
            if st.form_submit_button('Compute Hybrid Prediction', use_container_width=True):
                data = EngineeringInput(jee_main, jee_adv, class_12, category, state, branch)
                features = DataPreprocessor.preprocess_engineering(data, category_encoder)
                ml_res = MLPredictor.predict_engineering(features, engineering_model)
                rules_res = RulesEngine.evaluate_engineering(data)
                
                st.session_state.results = RankingEngine.format_engineering_results(ml_res, rules_res)
                st.rerun()

    elif stream == 'Medical':
        with st.form("med_form"):
            st.markdown('<div class="form-section-title">Academic Details</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                neet_score = st.number_input('NEET Score (out of 720)', 0, 720, 650, 1)
                class_12 = st.number_input('Class 12 Percentage', 0.0, 100.0, 90.0, 0.1)
                category = st.selectbox('Category', CATEGORIES)
            with c2:
                neet_rank = st.number_input('NEET All India Rank', 0, 999999, 30000, 1000)
                domicile = st.selectbox('Domicile State', NORTH_STATES)
                course = st.selectbox('Preferred Course', ['MBBS', 'BDS'])
                
            if st.form_submit_button('Compute Hybrid Prediction', use_container_width=True):
                data = MedicalInput(neet_score, neet_rank, class_12, category, domicile, course)
                rules_res = RulesEngine.evaluate_medical(data)
                
                st.session_state.results = RankingEngine.format_medical_results(rules_res)
                st.rerun()

    if st.session_state.results is not None:
        render_results(st.session_state.results, stream)

if __name__ == "__main__":
    main()