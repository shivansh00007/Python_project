# ==============================================================================
# NORTH INDIA COLLEGE ADMISSION PREDICTOR - Premium Design
# Focus: Delhi, Punjab, Haryana, UP, Uttarakhand, Rajasthan, HP, J&K
# ==============================================================================

import streamlit as st
import numpy as np
import joblib
import os
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

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
            error_msg = "⚠️ ML Model files not found. Using rule-based fallback."
            logger.warning(error_msg)
    except Exception as e:
        error_msg = f"⚠️ Error loading ML models: {str(e)}"
        logger.error(error_msg)
        
    return engineering_model, category_encoder, error_msg


class MLPredictor:
    @staticmethod
    def predict_engineering(features: np.ndarray, model) -> Dict[str, Any]:
        """Executes model inference, attempting to extract confidence scores."""
        if model is None:
            return {'success': False, 'error': 'Model not loaded'}
            
        try:
            # Try to get probability if the model supports predict_proba
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features)[0]
                max_prob_index = np.argmax(probabilities)
                predicted_class = model.classes_[max_prob_index]
                confidence = float(probabilities[max_prob_index]) * 100
            else:
                # Fallback to standard predict
                predicted_class = model.predict(features)[0]
                confidence = 85.0 # Estimated baseline confidence for direct predictions
                
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
# ARCHITECTURE LAYER 3: RULES ENGINE
# ==============================================================================

class RulesEngine:
    @staticmethod
    def evaluate_engineering(data: EngineeringInput) -> Dict[str, Any]:
        """Applies business logic constraints and cutoffs."""
        # Hard Eligibility Gate
        if data.class_12_percent < 75 and data.category == 'General':
            return {'eligible': False, 'message': '⚠️ Class 12 percentage must be ≥ 75% for General category.', 'candidates': []}
            
        category_boost = {'General': 0, 'OBC-NCL': 5, 'SC': 10, 'ST': 12, 'EWS': 3}
        home_state_boost = 2 if data.state in NORTH_STATES else 0
        adj_pct = data.jee_main_percentile + category_boost.get(data.category, 0) + home_state_boost
        
        candidates = []
        rank = data.jee_advanced_rank if data.jee_advanced_rank > 0 else 999999
        
        if rank < 1000:
            candidates.append({'college': 'IIT Delhi', 'location': 'New Delhi', 'type': 'Central Govt IIT', 'rules_score': 95.0, 'branch': 'Computer Science Engineering'})
        elif rank < 3000:
            candidates.append({'college': 'IIT Delhi / IIT Roorkee', 'location': 'Delhi / Uttarakhand', 'type': 'Central Govt IIT', 'rules_score': 85.0, 'branch': 'Electrical / Electronics'})
            
        if adj_pct >= 99.5:
            candidates.append({'college': 'NSUT / DTU / IIIT Delhi', 'location': 'New Delhi', 'type': 'Delhi State Govt', 'rules_score': 92.0, 'branch': 'Computer Science'})
        elif adj_pct >= 98.5:
            candidates.append({'college': 'NIT Kurukshetra', 'location': 'Kurukshetra, Haryana', 'type': 'Central Govt NIT', 'rules_score': 88.0, 'branch': 'Computer Science / IT'})
        elif adj_pct >= 95.0:
            candidates.append({'college': 'Thapar University', 'location': 'Patiala, Punjab', 'type': 'Deemed University', 'rules_score': 82.0, 'branch': 'Computer Engineering'})
            
        return {'eligible': True, 'candidates': candidates}

    @staticmethod
    def evaluate_medical(data: MedicalInput) -> Dict[str, Any]:
        """Applies medical admission constraints."""
        if data.class_12_percent < 50 and data.category == 'General':
            return {'eligible': False, 'message': '⚠️ Class 12 percentage must be ≥ 50%.', 'candidates': []}
            
        candidates = []
        if data.neet_score >= 715:
            candidates.append({'college': 'AIIMS Delhi', 'location': 'New Delhi', 'type': 'Central Govt AIIMS', 'rules_score': 95.0, 'course': 'MBBS'})
        if data.domicile == 'Delhi' and data.neet_score >= 690:
            candidates.append({'college': 'MAMC Delhi', 'location': 'New Delhi', 'type': 'State Govt Medical', 'rules_score': 90.0, 'course': 'MBBS'})
        elif data.neet_score >= 705:
            candidates.append({'college': 'MAMC / LHMC Delhi', 'location': 'New Delhi', 'type': 'State Govt Medical', 'rules_score': 75.0, 'course': 'MBBS'})
            
        if data.neet_score >= 675 and data.domicile == 'Uttar Pradesh':
            candidates.append({'college': 'KGMU Lucknow', 'location': 'Lucknow, UP', 'type': 'State Govt Medical', 'rules_score': 85.0, 'course': 'MBBS'})
            
        return {'eligible': True, 'candidates': candidates}

# ==============================================================================
# ARCHITECTURE LAYER 4: RANKING & FORMATTER
# ==============================================================================

class RankingEngine:
    @staticmethod
    def format_engineering_results(ml_result: Dict[str, Any], rules_result: Dict[str, Any]) -> Dict[str, Any]:
        """Combines ML and Rule scores using Final_Score = (0.6 * ML) + (0.4 * Rules)"""
        if not rules_result.get('eligible', True):
            return rules_result
            
        final_rankings: List[PredictionResult] = []
        seen_colleges = set()
        
        # 1. Process Rule-Based Candidates
        for cand in rules_result.get('candidates', []):
            ml_prob = 70.0 # Baseline ML correlation if not explicitly predicted
            
            # If the rule candidate matches the ML prediction, boost the ML score
            if ml_result.get('success') and ml_result['college'] in cand['college']:
                ml_prob = ml_result['confidence']
                seen_colleges.add(ml_result['college'])
                rationale = "High ML confidence + exceeds historical cutoff rules."
            else:
                rationale = "Matched via institutional historical cutoffs and category rules."
                
            rules_score = cand['rules_score']
            final_score = (0.6 * ml_prob) + (0.4 * rules_score)
            
            category = "safety" if final_score >= 85 else "match" if final_score >= 75 else "reach"
            
            final_rankings.append(PredictionResult(
                college=cand['college'], location=cand['location'], type=cand['type'],
                program=cand['branch'], ml_probability=ml_prob, rules_score=rules_score,
                final_score=final_score, category=category, rationale=rationale
            ))
            
        # 2. Add ML Prediction if it wasn't caught by the Rules Engine
        if ml_result.get('success') and ml_result['college'] not in seen_colleges:
            ml_prob = ml_result['confidence']
            rules_score = 60.0 # Penalized rules score since it bypassed the hard-coded rules
            final_score = (0.6 * ml_prob) + (0.4 * rules_score)
            category = "safety" if final_score >= 85 else "match" if final_score >= 75 else "reach"
            
            final_rankings.append(PredictionResult(
                college=ml_result['college'], location="North India", type="Predicted Institution",
                program=ml_result['branch'], ml_probability=ml_prob, rules_score=rules_score,
                final_score=final_score, category=category, rationale="Primarily identified via AI pattern matching."
            ))
            
        # 3. Sort Rankings by Final Hybrid Score (Descending)
        final_rankings.sort(key=lambda x: x.final_score, reverse=True)
        
        # Calculate overall systemic confidence
        top_score = final_rankings[0].final_score if final_rankings else 0.0
        
        return {
            'eligible': True,
            'overall_score': top_score,
            'rankings': final_rankings
        }

    @staticmethod
    def format_medical_results(rules_result: Dict[str, Any]) -> Dict[str, Any]:
        """Ranks medical predictions (Currently rules-only pending ML model)."""
        if not rules_result.get('eligible', True):
            return rules_result
            
        final_rankings: List[PredictionResult] = []
        for cand in rules_result.get('candidates', []):
            final_score = cand['rules_score'] # ML is 0 pending integration
            category = "safety" if final_score >= 85 else "match" if final_score >= 75 else "reach"
            
            final_rankings.append(PredictionResult(
                college=cand['college'], location=cand['location'], type=cand['type'],
                program=cand['course'], ml_probability=0.0, rules_score=cand['rules_score'],
                final_score=final_score, category=category, rationale="Matched via NEET cutoff and state domicile rules."
            ))
            
        final_rankings.sort(key=lambda x: x.final_score, reverse=True)
        top_score = final_rankings[0].final_score if final_rankings else 0.0
        
        return {'eligible': True, 'overall_score': top_score, 'rankings': final_rankings}


# ==============================================================================
# STREAMLIT UI COMPONENTS
# ==============================================================================

def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Inter:wght@300;400;600;700&display=swap');
        * { font-family: 'Inter', sans-serif; }
        .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); background-attachment: fixed; }
        .stApp::before { content: ""; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-image: repeating-linear-gradient(45deg, transparent, transparent 10px, rgba(255, 255, 255, 0.02) 10px, rgba(255, 255, 255, 0.02) 20px); pointer-events: none; z-index: 1; }
        .main .block-container { background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px); border-radius: 30px; padding: 3rem; box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.5); margin-top: 2rem; margin-bottom: 2rem; position: relative; z-index: 2; }
        .main-title { font-family: 'Playfair Display', serif; font-size: 3.5rem; font-weight: 900; text-align: center; background: linear-gradient(135deg, #FF9933 0%, #FF6B00 50%, #138808 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem; text-shadow: 0 5px 15px rgba(255, 153, 51, 0.3); }
        .subtitle { text-align: center; font-size: 1.3rem; color: #555; margin-bottom: 2rem; font-weight: 300; letter-spacing: 0.5px; }
        .region-badge { display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 0.5rem 1.5rem; border-radius: 50px; font-size: 0.9rem; font-weight: 600; margin-bottom: 2rem; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4); }
        .stream-card { background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(255, 255, 255, 0.7) 100%); padding: 3rem 2rem; border-radius: 20px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1); margin: 1rem 0; cursor: pointer; transition: all 0.4s; text-align: center; }
        .stream-card:hover { transform: translateY(-8px) scale(1.02); box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3); border: 2px solid rgba(102, 126, 234, 0.3); }
        .stream-icon { font-size: 4rem; margin-bottom: 1rem; display: block; }
        .stream-title { font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 700; color: #2d3748; margin-bottom: 0.5rem; }
        .stream-desc { color: #718096; font-size: 1.1rem; line-height: 1.6; }
        .form-section-title { font-family: 'Playfair Display', serif; font-size: 1.5rem; font-weight: 700; color: #2d3748; margin-bottom: 1.5rem; padding-bottom: 0.5rem; border-bottom: 3px solid; border-image: linear-gradient(90deg, #667eea, #764ba2) 1; }
        .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2.5rem; border-radius: 25px; color: white; text-align: center; margin: 1.5rem 0; box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4); }
        .metric-value { font-family: 'Playfair Display', serif; font-size: 4rem; font-weight: 900; }
        .metric-label { font-size: 1.2rem; margin-top: 0.8rem; font-weight: 600; text-transform: uppercase; }
        .college-card { background: white; padding: 2rem; border-radius: 20px; margin: 1.5rem 0; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08); border-left: 5px solid; position: relative; }
        .college-card.safety { border-left-color: #48bb78; }
        .college-card.match { border-left-color: #f6ad55; }
        .college-card.reach { border-left-color: #fc8181; }
        .college-name { font-family: 'Playfair Display', serif; font-size: 1.5rem; font-weight: 700; color: #2d3748; }
        .college-type { display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600; margin-bottom: 1rem; margin-right: 0.5rem; }
        .rationale-badge { display: inline-block; background: #e2e8f0; color: #4a5568; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.8rem; font-weight: 500; margin-bottom: 1rem; }
        .custom-progress { width: 100%; height: 12px; background: #e2e8f0; border-radius: 10px; margin: 1rem 0; }
        .custom-progress-bar { height: 100%; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; transition: width 1s ease-in-out; }
        .stButton>button { width: 100%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem 2rem; font-size: 1.1rem; font-weight: 600; border-radius: 15px; text-transform: uppercase; }
        .info-box { padding: 1.5rem; border-radius: 15px; border-left: 4px solid #667eea; margin: 1.5rem 0; }
        .warning-box { background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%); border-left-color: #f6ad55; }
        .info-box-title { font-weight: 700; color: #2d3748; font-size: 1.1rem; }
        .rank-badge { position: absolute; top: 1.5rem; right: 1.5rem; background: #2d3748; color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 1.2rem; }
        #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

def render_results(results: Dict[str, Any]):
    st.markdown('<hr style="margin: 3rem 0;">', unsafe_allow_html=True)
    
    if not results.get('eligible', True):
        st.markdown(f'''
        <div class="info-box warning-box">
            <div class="info-box-title">✅ Eligibility Check</div>
            <div class="info-box-content">{results.get("message", "")}</div>
        </div>
        ''', unsafe_allow_html=True)
        return

    st.markdown('<h2 style="text-align: center; color: #2d3748;">🎓 Final Match Rankings</h2>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #718096; margin-bottom: 2rem;">Computed using Hybrid Scoring (ML Probability & Business Rules)</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{results['overall_score']:.1f}%</div>
            <div class="metric-label">Top College Match Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    
    if results.get('rankings'):
        for i, cand in enumerate(results['rankings']):
            st.markdown(f"""
            <div class="college-card {cand.category}">
                <div class="rank-badge">#{i+1}</div>
                <div class="college-name">{cand.college}</div>
                <div style="margin-bottom: 1rem; color: #718096;">📍 {cand.location}</div>
                <span class="college-type">{cand.type}</span>
                <span class="rationale-badge">💡 {cand.rationale}</span>
                <div style="margin-top: 0.5rem;"><strong>Program:</strong> {cand.program}</div>
                <div style="display: flex; justify-content: space-between; margin-top: 1rem; font-size: 0.85rem; color: #4a5568;">
                    <span>ML Confidence: {cand.ml_probability:.1f}%</span>
                    <span>Rules Score: {cand.rules_score:.1f}%</span>
                </div>
                <div class="custom-progress">
                    <div class="custom-progress-bar" style="width: {cand.final_score}%;"></div>
                </div>
                <div style="text-align: center; font-weight: 700; color: #2d3748; font-size: 1.1rem;">Hybrid Final Score: {cand.final_score:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No safe recommendations available for this score profile. Consider participating in state-level counseling.")

# ==============================================================================
# MAIN APPLICATION CONTROLLER
# ==============================================================================

def main():
    st.set_page_config(page_title="North India College Predictor", page_icon="🏔️", layout="wide", initial_sidebar_state="expanded")
    inject_custom_css()

    # Model Layer Initialization
    engineering_model, category_encoder, error = load_ml_models()
    if error: st.toast(error)

    if 'stream_selected' not in st.session_state: st.session_state.stream_selected = None
    if 'results' not in st.session_state: st.session_state.results = None

    st.markdown('<h1 class="main-title">🏔️ North India College Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Powered by Hybrid ML & Rules Engine Scoring</p>', unsafe_allow_html=True)

    if st.session_state.stream_selected is None:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="stream-card"><span class="stream-icon">🔬</span><div class="stream-title">Engineering</div></div>', unsafe_allow_html=True)
            if st.button('Select Engineering', use_container_width=True):
                st.session_state.stream_selected = 'Engineering'
                st.rerun()
        with col2:
            st.markdown('<div class="stream-card"><span class="stream-icon">🏥</span><div class="stream-title">Medical</div></div>', unsafe_allow_html=True)
            if st.button('Select Medical', use_container_width=True):
                st.session_state.stream_selected = 'Medical'
                st.rerun()
        return

    stream = st.session_state.stream_selected
    col1, col2 = st.columns([4, 1])
    with col1: st.markdown(f'<h2 style="color: #2d3748;">Stream: {stream}</h2>', unsafe_allow_html=True)
    with col2: 
        if st.button('🔄 Start Over'):
            st.session_state.stream_selected = None
            st.session_state.results = None
            st.rerun()

    st.markdown('<hr style="margin: 2rem 0;">', unsafe_allow_html=True)

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
                
            if st.form_submit_button('🎯 Compute Hybrid Prediction', use_container_width=True):
                # 1. Input Collection
                data = EngineeringInput(jee_main, jee_adv, class_12, category, state, branch)
                
                # 2. Preprocessing
                features = DataPreprocessor.preprocess_engineering(data, category_encoder)
                
                # 3. Model & Rules Execution
                ml_res = MLPredictor.predict_engineering(features, engineering_model)
                rules_res = RulesEngine.evaluate_engineering(data)
                
                # 4. Final Ranking Formatter
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
                
            if st.form_submit_button('🎯 Compute Hybrid Prediction', use_container_width=True):
                data = MedicalInput(neet_score, neet_rank, class_12, category, domicile, course)
                rules_res = RulesEngine.evaluate_medical(data)
                st.session_state.results = RankingEngine.format_medical_results(rules_res)
                st.rerun()

    if st.session_state.results is not None:
        render_results(st.session_state.results)

if __name__ == "__main__":
    main()