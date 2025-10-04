import gradio as gr
import pandas as pd
import numpy as np
import pickle
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load the trained model
with open('rfcmodel (2).pkl', 'rb') as f:
    model = joblib.load(f)

# Feature descriptions for tooltips
feature_descriptions = {
    'person_education': 'Education level of the person (numeric scale)',
    'person_income': 'Annual income of the person',
    'person_emp_exp': 'Employment experience in years',
    'person_home_ownership': 'Home ownership status (numeric code)',
    'loan_amnt': 'Loan amount requested',
    'loan_intent': 'Loan purpose/intent (numeric code)',
    'loan_int_rate': 'Loan interest rate',
    'cb_person_cred_hist_length': 'Credit history length in years',
    'credit_score': 'Credit score of the applicant',
    'previous_loan_defaults_on_file': 'Number of previous loan defaults'
}

# Create prediction function
def predict_loan_risk(person_education, person_income, person_emp_exp, 
                     person_home_ownership, loan_amnt, loan_intent, 
                     loan_int_rate, cb_person_cred_hist_length, credit_score, 
                     previous_loan_defaults_on_file):
    
    # Create input array
    input_data = np.array([[person_education, person_income, person_emp_exp,
                          person_home_ownership, loan_amnt, loan_intent,
                          loan_int_rate, cb_person_cred_hist_length, credit_score,
                          previous_loan_defaults_on_file]])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]
    
    # Get class labels ( 0=Low Risk, 1=High Risk)
    risk_level = "Low Risk" if prediction == 0 else "High Risk"
    confidence = prediction_proba[prediction] * 100
    
    # Create detailed results
    result_html = f"""
    <div style="padding: 20px; border-radius: 10px; background: {'#d4edda' if prediction == 0 else '#f8d7da'}; border: 2px solid {'#c3e6cb' if prediction == 0 else '#f5c6cb'};">
        <h3 style="color: {'#155724' if prediction == 0 else '#721c24'}; margin-top: 0;">Loan Risk Assessment Result</h3>
        <div style="font-size: 24px; font-weight: bold; color: {'#155724' if prediction == 0 else '#721c24'};">{risk_level}</div>
        <div style="margin-top: 10px;">Confidence: <strong>{confidence:.2f}%</strong></div>
        <div style="margin-top: 15px; font-size: 14px;">
            Probability Breakdown:<br>
            - Low Risk: {prediction_proba[0]*100:.2f}%<br>
            - High Risk: {prediction_proba[1]*100:.2f}%
        </div>
    </div>
    """
    
    return result_html, risk_level, f"{confidence:.2f}%"

# Create feature inputs with better styling
def create_feature_inputs():
    inputs = []
    
    # Personal Information Section
    with gr.Group():
        gr.Markdown("### 👤 Personal Information")
        inputs.extend([
            gr.Number(label="Education Level", value=3, minimum=1, maximum=6, 
                     info="1: High School, 2: Some College, 3: Bachelor's, 4: Master's, 5: PhD, 6: Other"),
            gr.Number(label="Annual Income ($)", value=50000, minimum=0, maximum=1000000),
            gr.Number(label="Employment Experience (years)", value=5, minimum=0, maximum=50),
            gr.Number(label="Home Ownership", value=1, minimum=0, maximum=3,
                     info="0: Rent, 1: Mortgage, 2: Own, 3: Other")
        ])
    
    # Loan Information Section
    with gr.Group():
        gr.Markdown("### 💰 Loan Information")
        inputs.extend([
            gr.Number(label="Loan Amount ($)", value=10000, minimum=100, maximum=100000),
            gr.Number(label="Loan Intent", value=2, minimum=0, maximum=5,
                     info="0: Personal, 1: Education, 2: Medical, 3: Venture, 4: Home, 5: Other"),
            gr.Number(label="Interest Rate (%)", value=12.5, minimum=1.0, maximum=30.0, step=0.1)
        ])
    
    # Credit Information Section
    with gr.Group():
        gr.Markdown("### 💳 Credit Information")
        inputs.extend([
            gr.Number(label="Credit History Length (years)", value=8, minimum=0, maximum=50),
            gr.Number(label="Credit Score", value=650, minimum=300, maximum=850),
            gr.Number(label="Previous Loan Defaults", value=0, minimum=0, maximum=10)
        ])
    
    return inputs

# Custom CSS for beautiful styling
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 20px;
}
.result-box {
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
}
.low-risk {
    background: #d4edda;
    border: 2px solid #c3e6cb;
    color: #155724;
}
.high-risk {
    background: #f8d7da;
    border: 2px solid #f5c6cb;
    color: #721c24;
}
"""

# Create the Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    
    # Header
    gr.Markdown("""
    <div class="header">
        <h1>🏦 Smart Loan Risk Assessment</h1>
        <p>AI-powered loan risk prediction using Random Forest Classifier</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Input Parameters")
            inputs = create_feature_inputs()
            
            predict_btn = gr.Button("🔍 Assess Loan Risk", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            gr.Markdown("### 📈 Risk Assessment Results")
            result_html = gr.HTML()
            risk_level = gr.Textbox(label="Risk Level", interactive=False)
            confidence = gr.Textbox(label="Confidence Score", interactive=False)
            
            # Example values for quick testing
            gr.Markdown("### 💡 Quick Test Examples")
            with gr.Row():
                gr.Examples(
                    examples=[
                        [3, 50000, 5, 1, 10000, 2, 12.5, 8, 650, 0],  # Good applicant
                        [2, 25000, 1, 0, 20000, 0, 22.5, 2, 550, 3]   # Risky applicant
                    ],
                    inputs=inputs
                )
    
    # Footer
    gr.Markdown("""
    <div style="text-align: center; margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 10px;">
        <p><strong>Disclaimer:</strong> This is a demonstration tool. Actual loan decisions should be made by qualified financial professionals.</p>
        <p>Model: Random Forest Classifier | Features: 10 | Training: Comprehensive credit data</p>
    </div>
    """)
    
    # Connect the prediction function
    predict_btn.click(
        fn=predict_loan_risk,
        inputs=inputs,
        outputs=[result_html, risk_level, confidence]
    )

# Launch the application
if __name__ == "__main__":
    demo.launch()