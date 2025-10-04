🏦 Smart Loan Risk Assessment

A machine learning project that predicts loan approval risk using Random Forest Classifier with an interactive web interface built using Gradio.

📋 Project Overview

This project implements an AI-powered loan risk assessment system that analyzes various applicant features to predict whether a loan application is low-risk or high-risk. The system uses a Random Forest Classifier trained on loan approval data and provides an intuitive web interface for real-time predictions.

🗂️ Project Structure

├── Loan.ipynb              📊 Jupyter notebook with data analysis and model training
├── rfc.py                  🌐 Gradio web application for loan risk prediction
├── rfcmodel (2).pkl        🤖 Trained Random Forest model (serialized)
├── .gradio/
│   └── certificate.pem     🔒 SSL certificate for secure connections
└── README.md               📖 Project documentation

✨ Features

- 📊 Data Analysis & Preprocessing: Complete exploratory data analysis with visualization
- 🤖 Machine Learning Model: Random Forest Classifier with cross-validation
- 🌐 Interactive Web Interface: User-friendly Gradio app with real-time predictions
- ⚖️ Risk Assessment: Provides risk level classification with confidence scores
- ✅ Comprehensive Input Validation: Handles various loan and applicant parameters

📈 Model Features

The model uses the following 10 features for prediction:

1. 🎓 Person Education - Education level (1-6 scale)
2. 💰 Person Income - Annual income in dollars
3. 👔 Person Employment Experience - Years of employment experience
4. 🏠 Person Home Ownership - Home ownership status (0-3 coded)
5. 💵 Loan Amount - Requested loan amount
6. 🎯 Loan Intent - Purpose of the loan (0-5 coded)
7. 📊 Loan Interest Rate - Interest rate percentage
8. 📅 Credit History Length - Years of credit history
9. 💳 Credit Score - Credit score (300-850)
10. ⚠️ Previous Loan Defaults - Number of previous defaults

🚀 Getting Started

Prerequisites

pip install pandas numpy matplotlib seaborn scikit-learn gradio joblib kagglehub

Installation

1. 📥 Clone or download this repository
2. 🔧 Install the required dependencies
3. 📁 Ensure the model file rfcmodel (2).pkl is in the project directory

🎮 Running the Application

Option 1: Web Interface (Recommended) 🌟
python rfc.py

This will launch the Gradio web interface where you can:
- 📝 Input loan applicant details through an intuitive form
- ⚡ Get instant risk predictions with confidence scores
- 📊 View probability breakdowns for both risk levels
- 🧪 Test with pre-configured examples

Option 2: Jupyter Notebook 📓
jupyter notebook Loan.ipynb

Use this for:
- 🔍 Data exploration and analysis
- 🏋️ Model training and evaluation
- 🧠 Understanding the machine learning pipeline

🎯 Model Performance

- 🤖 Algorithm: Random Forest Classifier
- ✅ Cross-validation: 10-fold cross-validation implemented
- 📊 Features: 10 carefully selected features
- 📚 Data Source: Kaggle loan approval classification dataset
- 🔧 Preprocessing: Handles missing values, categorical encoding, and feature scaling

🖥️ Web Interface Features

The Gradio application includes:

📋 Organized Input Sections:
  - 👤 Personal Information
  - 💰 Loan Information  
  - 💳 Credit Information

📈 Real-time Results:
  - 🚦 Risk level classification (Low Risk/High Risk)
  - 📊 Confidence percentage
  - 📉 Probability breakdown
  - 🎨 Color-coded results for easy interpretation

🎨 User Experience:
  - 📱 Responsive design with custom CSS
  - ✅ Input validation and helpful tooltips
  - 🧪 Quick test examples for demonstration
  - 🎭 Professional styling with gradient themes

💡 Usage Examples

✅ Low Risk Example:
- 🎓 Education: Bachelor's (3)
- 💰 Income: $50,000
- 👔 Experience: 5 years
- 🏠 Home: Mortgage (1)
- 💵 Loan Amount: $10,000
- 🏥 Intent: Medical (2)
- 📊 Interest Rate: 12.5%
- 📅 Credit History: 8 years
- 💳 Credit Score: 650
- ⚠️ Previous Defaults: 0

❌ High Risk Example:
- 🎓 Education: High School (2)
- 💰 Income: $25,000
- 👔 Experience: 1 year
- 🏠 Home: Rent (0)
- 💵 Loan Amount: $20,000
- 🛍️ Intent: Personal (0)
- 📊 Interest Rate: 22.5%
- 📅 Credit History: 2 years
- 💳 Credit Score: 550
- ⚠️ Previous Defaults: 3

🔒 Security

- 🛡️ SSL certificate included for secure HTTPS connections
- ✅ Input validation to prevent malicious data
- 🔐 Model serialization using joblib for security

⚠️ Disclaimer

This is a demonstration tool for educational and research purposes. Actual loan decisions should always be made by qualified financial professionals considering additional factors, regulations, and comprehensive risk assessment procedures.

🛠️ Technical Details

- 🌐 Framework: Gradio for web interface
- 🤖 ML Library: scikit-learn
- 📊 Data Processing: pandas, numpy
- 📈 Visualization: matplotlib, seaborn
- 💾 Model Persistence: joblib
- 📚 Data Source: Kaggle via kagglehub

🚀 Future Enhancements

- 🔗 Integration with real-time credit scoring APIs
- 🤖 Additional model algorithms for comparison
- 📊 Enhanced data visualization dashboard
- 📦 Batch processing capabilities
- 🔄 Model retraining pipeline
- ⚙️ Advanced feature engineering

🤝 Contributing

Feel free to fork this project and submit pull requests for improvements. Areas for contribution include:
- 🚀 Model performance optimization
- 🎨 UI/UX enhancements
- ⚙️ Additional feature engineering
- 📖 Documentation improvements

Built with ❤️ using Python, scikit-learn, and Gradio
