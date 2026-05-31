# ==============================================================================
# NORTH INDIA COLLEGE ADMISSION PREDICTOR - Premium Design
# Focus: Delhi, Punjab, Haryana, UP, Uttarakhand, Rajasthan, HP, J&K
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="North India College Predictor | Engineering & Medical",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# PREMIUM CSS STYLING
# ==============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Inter:wght@300;400;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background-image: repeating-linear-gradient(45deg, transparent, transparent 10px, rgba(255, 255, 255, 0.02) 10px, rgba(255, 255, 255, 0.02) 20px);
        pointer-events: none;
        z-index: 1;
    }
    
    .main .block-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 30px;
        padding: 3rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.5);
        margin-top: 2rem; margin-bottom: 2rem;
        position: relative; z-index: 2;
    }
    
    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 3.5rem; font-weight: 900; text-align: center;
        background: linear-gradient(135deg, #FF9933 0%, #FF6B00 50%, #138808 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem; text-shadow: 0 5px 15px rgba(255, 153, 51, 0.3);
    }
    
    .subtitle {
        text-align: center; font-size: 1.3rem; color: #555;
        margin-bottom: 2rem; font-weight: 300; letter-spacing: 0.5px;
    }
    
    .region-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 0.5rem 1.5rem; border-radius: 50px;
        font-size: 0.9rem; font-weight: 600; margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stream-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(255, 255, 255, 0.7) 100%);
        padding: 3rem 2rem; border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1); margin: 1rem 0;
        cursor: pointer; transition: all 0.4s; text-align: center;
    }
    
    .stream-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        border: 2px solid rgba(102, 126, 234, 0.3);
    }
    
    .stream-icon { font-size: 4rem; margin-bottom: 1rem; display: block; }
    .stream-title { font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 700; color: #2d3748; margin-bottom: 0.5rem; }
    .stream-desc { color: #718096; font-size: 1.1rem; line-height: 1.6; }
    
    .form-section-title {
        font-family: 'Playfair Display', serif; font-size: 1.5rem; font-weight: 700;
        color: #2d3748; margin-bottom: 1.5rem; padding-bottom: 0.5rem;
        border-bottom: 3px solid; border-image: linear-gradient(90deg, #667eea, #764ba2) 1;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem; border-radius: 25px; color: white; text-align: center;
        margin: 1.5rem 0; box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
    }
    
    .metric-value { font-family: 'Playfair Display', serif; font-size: 4rem; font-weight: 900; }
    .metric-label { font-size: 1.2rem; margin-top: 0.8rem; font-weight: 600; text-transform: uppercase; }
    
    .college-card {
        background: white; padding: 2rem; border-radius: 20px;
        margin: 1.5rem 0; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08); border-left: 5px solid;
    }
    .college-card.safety { border-left-color: #48bb78; }
    .college-card.match { border-left-color: #f6ad55; }
    .college-card.reach { border-left-color: #fc8181; }
    
    .college-name { font-family: 'Playfair Display', serif; font-size: 1.5rem; font-weight: 700; color: #2d3748; }
    .college-type {
        display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600; margin-bottom: 1rem;
    }
    
    .custom-progress { width: 100%; height: 12px; background: #e2e8f0; border-radius: 10px; margin: 1rem 0; }
    .custom-progress-bar { height: 100%; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; }
    
    .stButton>button {
        width: 100%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 1rem 2rem; font-size: 1.1rem; font-weight: 600; border-radius: 15px; text-transform: uppercase;
    }
    
    .info-box { padding: 1.5rem; border-radius: 15px; border-left: 4px solid #667eea; margin: 1.5rem 0; }
    .success-box { background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%); border-left-color: #48bb78; }
    .warning-box { background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%); border-left-color: #f6ad55; }
    .info-box-title { font-weight: 700; color: #2d3748; font-size: 1.1rem; }
    
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# ML MODEL LOADING - ENGINEERING
# ==============================================================================

engineering_model = None
category_encoder = None
model_load_error = None

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'engineering_model_north.pkl')
    encoder_path = os.path.join(current_dir, 'category_encoder_north.pkl')
    
    if os.path.exists(model_path) and os.path.exists(encoder_path):
        engineering_model = joblib.load(model_path)
        category_encoder = joblib.load(encoder_path)
    else:
        model_load_error = "⚠️ ML Model files not found. Using rule-based fallback."
except Exception as e:
    model_load_error = f"⚠️ Error loading ML models: {str(e)}"

# ==============================================================================
# ML PREDICTION FUNCTION - ENGINEERING
# ==============================================================================

def predict_engineering(percentile, rank, category):
    try:
        if rank is None or rank == 0 or rank == 999999:
            rank = int((1 - percentile / 100) * 1400000)
        
        if category_encoder is not None:
            try:
                category_encoded = category_encoder.transform([category])[0]
            except:
                category_map = {'General': 0, 'OBC-NCL': 1, 'SC': 2, 'ST': 3, 'EWS': 4}
                category_encoded = category_map.get(category, 0)
        else:
            category_map = {'General': 0, 'OBC-NCL': 1, 'SC': 2, 'ST': 3, 'EWS': 4}
            category_encoded = category_map.get(category, 0)
        
        features = np.array([[rank, category_encoded]])
        
        if engineering_model is not None:
            prediction = engineering_model.predict(features)
            predicted_class = prediction[0] if len(prediction) > 0 else None
            
            if predicted_class is not None:
                prediction_str = str(predicted_class)
                if " - " in prediction_str:
                    college, branch = prediction_str.split(" - ", 1)
                else:
                    college = prediction_str
                    branch = "Engineering"
                
                return {'college': college, 'branch': branch, 'error': None}
        return {'error': 'ML model not loaded'}
    except Exception as e:
        return {'error': f'Prediction error: {str(e)}'}

# ==============================================================================
# RULE-BASED PREDICTION FUNCTIONS
# ==============================================================================

def predict_north_engineering(data):
    jee_percentile = data['jee_main_percentile']
    jee_adv_rank = data.get('jee_advanced_rank', 999999)
    class_12 = data['class_12_percent']
    category = data['category']
    state = data['state']
    
    category_boost = {'General': 0, 'OBC-NCL': 5, 'SC': 10, 'ST': 12, 'EWS': 3}
    home_state_boost = 2 if state in ['Delhi', 'Haryana', 'Punjab', 'UP', 'Uttarakhand', 'Rajasthan'] else 0
    adjusted_percentile = jee_percentile + category_boost.get(category, 0) + home_state_boost
    
    predictions = []
    
    if jee_adv_rank < 999999:
        if jee_adv_rank < 1000: predictions.append({'college': 'IIT Delhi', 'location': 'New Delhi', 'type': 'Central Govt IIT', 'probability': 90, 'category': 'safety', 'branch': 'Computer Science Engineering'})
        elif jee_adv_rank < 3000: predictions.append({'college': 'IIT Delhi / IIT Roorkee', 'location': 'Delhi / Uttarakhand', 'type': 'Central Govt IIT', 'probability': 75, 'category': 'match', 'branch': 'Electrical / Electronics'})
    
    if adjusted_percentile >= 99.5: predictions.append({'college': 'NSUT / DTU / IIIT Delhi', 'location': 'New Delhi', 'type': 'Delhi State Govt', 'probability': 85, 'category': 'safety', 'branch': 'Computer Science'})
    elif adjusted_percentile >= 98.5: predictions.append({'college': 'NIT Kurukshetra', 'location': 'Kurukshetra, Haryana', 'type': 'Central Govt NIT', 'probability': 85, 'category': 'safety', 'branch': 'Computer Science / IT'})
    
    if adjusted_percentile >= 95.0: predictions.append({'college': 'Thapar University', 'location': 'Patiala, Punjab', 'type': 'Deemed University', 'probability': 90, 'category': 'safety', 'branch': 'Computer Engineering'})
    
    if class_12 < 75 and category == 'General':
        return {'eligible': False, 'message': '⚠️ Class 12 percentage should be 75% or above for General category', 'predictions': []}
    
    return {'eligible': True, 'overall_score': (jee_percentile + class_12) / 2, 'predictions': predictions}

def predict_north_medical(data):
    neet_score = data['neet_score']
    class_12 = data['class_12_percent']
    domicile = data['domicile']
    predictions = []
    
    if neet_score >= 715: predictions.append({'college': 'AIIMS Delhi', 'location': 'New Delhi', 'type': 'Central Govt AIIMS', 'probability': 80, 'category': 'match', 'course': 'MBBS'})
    if domicile == 'Delhi' and neet_score >= 690: predictions.append({'college': 'MAMC Delhi', 'location': 'New Delhi', 'type': 'State Govt Medical', 'probability': 90, 'category': 'safety', 'course': 'MBBS'})
    elif neet_score >= 705: predictions.append({'college': 'MAMC / LHMC Delhi', 'location': 'New Delhi', 'type': 'State Govt Medical', 'probability': 70, 'category': 'reach', 'course': 'MBBS'})
    
    if neet_score >= 675 and domicile == 'UP': predictions.append({'college': 'KGMU Lucknow', 'location': 'Lucknow, UP', 'type': 'State Govt Medical', 'probability': 85, 'category': 'safety', 'course': 'MBBS'})
    
    if class_12 < 50 and data['category'] == 'General':
        return {'eligible': False, 'message': '⚠️ Class 12 percentage should be 50% or above', 'predictions': []}
    
    return {'eligible': True, 'overall_score': (neet_score / 720) * 100, 'predictions': predictions}

# ==============================================================================
# UI FLOW
# ==============================================================================

if 'stream_selected' not in st.session_state:
    st.session_state.stream_selected = None
if 'results' not in st.session_state:
    st.session_state.results = None

if model_load_error:
    st.toast(model_load_error)

st.markdown('<h1 class="main-title">🏔️ North India College Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Engineering (IITs, NITs) • Medical (AIIMS, State Medical)</p>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center;"><span class="region-badge">📍 Delhi, Punjab, Haryana, UP, Uttarakhand, Rajasthan</span></div><br>', unsafe_allow_html=True)

if st.session_state.stream_selected is None:
    st.markdown('<h2 style="text-align: center; color: #2d3748; margin-bottom: 2rem;">Choose Your Stream</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="stream-card">
            <span class="stream-icon">🔬</span>
            <div class="stream-title">Engineering (PCM)</div>
            <div class="stream-desc">IIT Delhi, NIT Kurukshetra, NSUT, DTU, IIIT Delhi</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button('Select Engineering', use_container_width=True):
            st.session_state.stream_selected = 'PCM'
            st.rerun()
            
    with col2:
        st.markdown("""
        <div class="stream-card">
            <span class="stream-icon">🏥</span>
            <div class="stream-title">Medical (PCB)</div>
            <div class="stream-desc">AIIMS Delhi, MAMC, LHMC, KGMU Lucknow, PGI Chandigarh</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button('Select Medical', use_container_width=True):
            st.session_state.stream_selected = 'PCB'
            st.rerun()

else:
    stream = st.session_state.stream_selected
    col1, col2 = st.columns([4, 1])
    with col1: st.markdown(f'<h2 style="color: #2d3748;">Selected: {stream}</h2>', unsafe_allow_html=True)
    with col2: 
        if st.button('🔄 Start Over'):
            st.session_state.stream_selected = None
            st.session_state.results = None
            st.rerun()
            
    st.markdown('<hr style="margin: 2rem 0;">', unsafe_allow_html=True)
    north_states = ['Delhi', 'Punjab', 'Haryana', 'Uttar Pradesh', 'Uttarakhand', 'Rajasthan', 'Himachal Pradesh', 'Jammu & Kashmir']
    categories = ['General', 'OBC-NCL', 'SC', 'ST', 'EWS']

    if stream == 'PCM':
        with st.form("pcm_form"):
            st.markdown('<div class="form-section-title">Academic & Personal Details</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                jee_main_percentile = st.number_input('JEE Main Percentile', 0.0, 100.0, 95.0, 0.1)
                class_12_percent = st.number_input('Class 12 Percentage', 0.0, 100.0, 90.0, 0.1)
                category_sel = st.selectbox('Category', categories)
            with col2:
                jee_adv_rank = st.number_input('JEE Advanced Rank (0 if N/A)', 0, 999999, 0)
                state_sel = st.selectbox('Home State', north_states)
                branch = st.selectbox('Preferred Branch', ['Computer Science', 'Electronics', 'Electrical', 'Mechanical'])
                
            if st.form_submit_button('🎯 Calculate My Chances', use_container_width=True):
                ml_prediction = predict_engineering(jee_main_percentile, jee_adv_rank, category_sel)
                data = {'jee_main_percentile': jee_main_percentile, 'jee_advanced_rank': jee_adv_rank if jee_adv_rank > 0 else 999999, 'class_12_percent': class_12_percent, 'category': category_sel, 'state': state_sel}
                results = predict_north_engineering(data)
                results['ml_prediction'] = ml_prediction
                st.session_state.results = results
                st.rerun()

    elif stream == 'PCB':
        with st.form("pcb_form"):
            st.markdown('<div class="form-section-title">Academic & Personal Details</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                neet_score = st.number_input('NEET Score (out of 720)', 0, 720, 650, 1)
                class_12_percent = st.number_input('Class 12 Percentage', 0.0, 100.0, 90.0, 0.1)
                category_sel = st.selectbox('Category', categories)
            with col2:
                neet_rank = st.number_input('NEET All India Rank', 0, 999999, 30000, 1000)
                domicile = st.selectbox('Domicile State', north_states)
                course = st.selectbox('Preferred Course', ['MBBS', 'BDS'])
                
            if st.form_submit_button('🎯 Calculate My Chances', use_container_width=True):
                data = {'neet_score': neet_score, 'class_12_percent': class_12_percent, 'category': category_sel, 'domicile': domicile}
                results = predict_north_medical(data)
                st.session_state.results = results
                st.rerun()

if st.session_state.results is not None:
    st.markdown('<hr style="margin: 3rem 0;">', unsafe_allow_html=True)
    results = st.session_state.results
    
    if not results.get('eligible', True):
        st.markdown(f'<div class="info-box warning-box"><div class="info-box-title">✅ Eligibility Check</div><div class="info-box-content">{results.get("message", "")}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<h2 style="text-align: center; color: #2d3748;">🎓 Your Admission Prediction</h2>', unsafe_allow_html=True)
        
        if st.session_state.stream_selected == 'PCM' and 'ml_prediction' in results:
            ml_pred = results['ml_prediction']
            if ml_pred.get('error') is None:
                st.markdown(f"""
                <div class="info-box success-box">
                    <div class="info-box-title">🤖 AI Machine Learning Prediction</div>
                    <div class="info-box-content">
                        <strong>Predicted College:</strong> {ml_pred['college']}<br>
                        <strong>Predicted Branch:</strong> {ml_pred['branch']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{results['overall_score']:.1f}%</div>
                <div class="metric-label">Overall Success Probability</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<h3 style="color: #2d3748; margin-top: 2rem;">🏛️ Recommended Colleges</h3>', unsafe_allow_html=True)
        
        if results['predictions']:
            for pred in results['predictions']:
                branch_or_course = pred.get('branch', pred.get('course', 'N/A'))
                st.markdown(f"""
                <div class="college-card {pred['category']}">
                    <div class="college-name">{pred['college']}</div>
                    <div class="college-location">📍 {pred['location']}</div>
                    <span class="college-type">{pred['type']}</span>
                    <div style="margin-top: 1rem;"><strong>Program:</strong> {branch_or_course}</div>
                    <div class="custom-progress" style="margin-top: 1rem;">
                        <div class="custom-progress-bar" style="width: {pred['probability']}%;"></div>
                    </div>
                    <div style="text-align: center; font-weight: 600; color: #4a5568;">{pred['probability']}% Probability</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No safe recommendations available for this score in top-tier colleges. Consider state-level counseling.")