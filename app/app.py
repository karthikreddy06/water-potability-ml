"""
Water Potability Prediction - Streamlit Web Application
This app allows users to input water quality features and predict potability.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Water Potability Predictor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# STYLING
# ============================================================================

st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .stContainer {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: #0066cc;
            text-align: center;
            margin-bottom: 2rem;
        }
        .prediction-safe {
            padding: 1.5rem;
            border-radius: 0.5rem;
            background-color: #d4edda;
            border: 2px solid #28a745;
            color: #155724;
        }
        .prediction-unsafe {
            padding: 1.5rem;
            border-radius: 0.5rem;
            background-color: #f8d7da;
            border: 2px solid #dc3545;
            color: #721c24;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL AND DATA LOADING
# ============================================================================

@st.cache_resource
def load_model_and_scaler():
    """Load the trained model, scaler, and feature names."""
    models_dir = 'models/'
    
    # Find the most recent model file
    model_files = sorted(Path(models_dir).glob('*.pkl'))
    model_files = [f for f in model_files if 'random_forest' in f.name or 'xgboost' in f.name 
                   or 'k_nearest_neighbors' in f.name]
    
    # Try to find h5 file (neural network)
    h5_files = sorted(Path(models_dir).glob('*.h5'))
    
    if model_files:
        model_path = model_files[-1]
        model = joblib.load(model_path)
        model_name = model_path.stem.replace('_', ' ').title()
    elif h5_files:
        import tensorflow.keras as keras
        model_path = h5_files[-1]
        model = keras.models.load_model(model_path)
        model_name = "Neural Network"
    else:
        st.error("❌ No trained model found. Please run train.py first.")
        st.stop()
    
    # Load scaler
    scaler_files = sorted(Path(models_dir).glob('scaler*.pkl'))
    if scaler_files:
        scaler = joblib.load(scaler_files[-1])
    else:
        scaler = None
    
    # Load feature names
    feature_names_path = Path(models_dir) / 'feature_names.pkl'
    if feature_names_path.exists():
        feature_names = joblib.load(feature_names_path)
    else:
        feature_names = ['pH', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                        'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
    
    return model, scaler, feature_names, model_name

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Title
    st.title("💧 Water Potability Prediction System")
    
    st.markdown("""
    This application predicts whether drinking water is **potable (safe)** or **non-potable (unsafe)**
    based on water quality features. Enter the water quality measurements below to get a prediction.
    """)
    
    # Load model and scaler
    with st.spinner("Loading model and scaler..."):
        try:
            model, scaler, feature_names, model_name = load_model_and_scaler()
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.stop()
    
    # Display model information
    with st.sidebar:
        st.header("ℹ️ Model Information")
        st.info(f"**Active Model:** {model_name}")
        st.markdown("""
        ### Feature Descriptions:
        - **pH**: Acidity/basicity (0-14)
        - **Hardness**: Mineral content (mg/L)
        - **Solids**: Total dissolved solids (mg/L)
        - **Chloramines**: Disinfectant level (mg/L)
        - **Sulfate**: Sulfate concentration (mg/L)
        - **Conductivity**: Electrical conductivity (µS/cm)
        - **Organic Carbon**: Organic matter (mg/L)
        - **Trihalomethanes**: Disinfection byproducts (µg/L)
        - **Turbidity**: Water clarity (NTU)
        """)
    
    # Input section
    st.header("📊 Water Quality Measurements")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pH = st.number_input(
            "pH",
            min_value=0.0,
            max_value=14.0,
            value=7.0,
            step=0.1,
            help="Acidity level (0-14)"
        )
        Hardness = st.number_input(
            "Hardness (mg/L)",
            min_value=0.0,
            max_value=500.0,
            value=50.0,
            step=1.0,
            help="Mineral content"
        )
        Solids = st.number_input(
            "Solids (mg/L)",
            min_value=0.0,
            max_value=100000.0,
            value=20000.0,
            step=100.0,
            help="Total dissolved solids"
        )
    
    with col2:
        Chloramines = st.number_input(
            "Chloramines (mg/L)",
            min_value=0.0,
            max_value=10.0,
            value=4.0,
            step=0.1,
            help="Disinfectant level"
        )
        Sulfate = st.number_input(
            "Sulfate (mg/L)",
            min_value=0.0,
            max_value=500.0,
            value=150.0,
            step=1.0,
            help="Sulfate concentration"
        )
        Conductivity = st.number_input(
            "Conductivity (µS/cm)",
            min_value=0.0,
            max_value=1000.0,
            value=300.0,
            step=1.0,
            help="Electrical conductivity"
        )
    
    with col3:
        Organic_carbon = st.number_input(
            "Organic Carbon (mg/L)",
            min_value=0.0,
            max_value=20.0,
            value=2.5,
            step=0.1,
            help="Organic matter content"
        )
        Trihalomethanes = st.number_input(
            "Trihalomethanes (µg/L)",
            min_value=0.0,
            max_value=100.0,
            value=40.0,
            step=1.0,
            help="Disinfection byproducts"
        )
        Turbidity = st.number_input(
            "Turbidity (NTU)",
            min_value=0.0,
            max_value=10.0,
            value=3.5,
            step=0.1,
            help="Water clarity"
        )
    
    # Prediction
    if st.button("🔮 Predict Potability", key="predict_btn", use_container_width=True):
        # Prepare input data
        input_data = pd.DataFrame({
            'pH': [pH],
            'Hardness': [Hardness],
            'Solids': [Solids],
            'Chloramines': [Chloramines],
            'Sulfate': [Sulfate],
            'Conductivity': [Conductivity],
            'Organic_carbon': [Organic_carbon],
            'Trihalomethanes': [Trihalomethanes],
            'Turbidity': [Turbidity]
        })
        
        # Ensure column order matches training data
        input_data = input_data[feature_names]
        
        # Scale input
        if scaler is not None:
            input_scaled = scaler.transform(input_data)
            input_data = pd.DataFrame(input_scaled, columns=feature_names)
        
        # Make prediction
        try:
            # Handle different model types
            if hasattr(model, 'predict_proba'):
                # Sklearn models
                prediction = model.predict(input_data)[0]
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(input_data)[0]
                    confidence = max(probabilities) * 100
                else:
                    confidence = 0
            else:
                # Neural Network
                import tensorflow as tf
                prediction_prob = model.predict(input_data, verbose=0)[0][0]
                prediction = 1 if prediction_prob > 0.5 else 0
                confidence = (prediction_prob * 100 if prediction == 1 
                            else (1 - prediction_prob) * 100)
            
            # Display result
            st.markdown("---")
            
            if prediction == 1:
                st.markdown(
                    f"""
                    <div class="prediction-safe">
                        <h2 style="margin: 0; font-size: 2em;">✅ POTABLE (SAFE)</h2>
                        <p style="margin: 0.5rem 0 0 0; font-size: 1.2em;">
                            Confidence: <strong>{confidence:.1f}%</strong>
                        </p>
                        <p style="margin-top: 0.5rem;">
                            This water appears to be safe for drinking based on the provided measurements.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="prediction-unsafe">
                        <h2 style="margin: 0; font-size: 2em;">⚠️ NON-POTABLE (UNSAFE)</h2>
                        <p style="margin: 0.5rem 0 0 0; font-size: 1.2em;">
                            Confidence: <strong>{confidence:.1f}%</strong>
                        </p>
                        <p style="margin-top: 0.5rem;">
                            This water does not appear to be safe for drinking. Further treatment is recommended.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Display input summary
            st.markdown("---")
            st.subheader("📋 Input Summary")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("pH", f"{pH:.2f}")
                st.metric("Hardness", f"{Hardness:.1f} mg/L")
                st.metric("Solids", f"{Solids:.0f} mg/L")
            with col2:
                st.metric("Chloramines", f"{Chloramines:.2f} mg/L")
                st.metric("Sulfate", f"{Sulfate:.1f} mg/L")
                st.metric("Conductivity", f"{Conductivity:.1f} µS/cm")
            with col3:
                st.metric("Organic Carbon", f"{Organic_carbon:.2f} mg/L")
                st.metric("Trihalomethanes", f"{Trihalomethanes:.1f} µg/L")
                st.metric("Turbidity", f"{Turbidity:.2f} NTU")
        
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray; margin-top: 2rem;">
        <p><strong>Water Potability Prediction System</strong></p>
        <p>Built with Streamlit | Powered by Machine Learning</p>
        <p><em>For demonstration purposes only. Always consult official water quality reports.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
