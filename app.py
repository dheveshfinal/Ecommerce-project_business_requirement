import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(
    page_title="E-commerce Customer Spending Predictor",
    page_icon="🛒",
    layout="wide"
)

@st.cache_resource
def load_model():
    with open("ridge_model.pkl", "rb") as model_file:
        return pickle.load(model_file)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("C:/Users/dheve/OneDrive/Desktop/ML-project/Ecommerce project_business_requirement/Ecommerce_Customers.csv")

try:
    model = load_model()
    df = load_data()
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

# Custom CSS with reduced font sizes
st.markdown("""
    <style>
        .main {background-color: #f8f9fa;}
        .block-container {padding-top: 1rem;}
        h1 {font-size: 28px !important;}
        h2 {font-size: 22px !important;}
        h3 {font-size: 18px !important;}
        p, li, div {font-size: 14px !important;}
        .input-section {
            background-color: #f1f8ff;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        .prediction-box {
            background-color: #e8f5e9;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
        }
        .viz-section {
            background-color: white;
            padding: 12px;
            border-radius: 5px;
            margin: 12px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .big-number {
            font-size: 28px !important;
            font-weight: bold;
            color: #2e7d32;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 30px;
            padding-top: 4px;
            font-size: 14px !important;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🛒 E-commerce Customer Spending Predictor")
st.write("Predict the yearly amount spent based on customer behavior.")

# Information expander
with st.expander("ℹ️ About This Predictor"):
    st.write("""
    This tool uses machine learning to predict a customer's yearly spending based on their behavior patterns.
    The Ridge Regression model was trained on historical customer data with the following features:
    - **Length of Membership**: How long the customer has been with the company
    - **Time on App**: Average time spent on the company's mobile application
    - **Average Session Length**: How long a typical user session lasts
    """)

# Top section for inputs and prediction - full width
st.markdown("## 📝 Enter Customer Data")

# Create 3 columns for inputs
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("### Length of Membership (Years)")
    length_of_membership = st.number_input("Enter value", min_value=0.5, max_value=10.0, value=5.0, step=0.1, key="membership")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("### Time on App (Minutes)")
    time_on_app = st.number_input("Enter minutes", min_value=5.0, max_value=20.0, value=12.0, step=0.5, key="app_time")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("### Avg Session Length (Minutes)")
    avg_session_length = st.number_input("Enter average", min_value=20.0, max_value=50.0, value=30.0, step=1.0, key="session_length")
    st.markdown('</div>', unsafe_allow_html=True)

# Prediction Section
# Prepare input for model and predict
input_features = np.array([[length_of_membership, time_on_app, avg_session_length]])
predicted_amount = model.predict(input_features)[0]

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
    st.markdown("## 💰 Predicted Yearly Spending")
    st.markdown(f'<p class="big-number">${predicted_amount:.2f}</p>', unsafe_allow_html=True)
    
    # Add CSV download button
    @st.cache_data
    def generate_csv():
        data = {
            'Feature': ['Length of Membership', 'Time on App', 'Avg Session Length', 'Predicted Spending'],
            'Value': [length_of_membership, time_on_app, avg_session_length, predicted_amount]
        }
        return pd.DataFrame(data)
    
    csv = generate_csv()
    st.download_button(
        label="Download Customer Profile (CSV)",
        data=csv.to_csv(index=False).encode('utf-8'),
        file_name='customer_profile.csv',
        mime='text/csv',
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Feature importance - horizontal bar chart
    st.markdown("## 📊 Feature Impact")
    feature_names = ['Membership Length', 'App Time', 'Session Length']
    feature_impacts = [0.65, 0.20, 0.15]  # Example values
    
    fig_impact = plt.figure(figsize=(6, 2.5))
    plt.barh(feature_names, feature_impacts, color=['#2e7d32', '#388e3c', '#43a047'])
    plt.xlabel('Impact on Spending')
    plt.tight_layout()
    st.pyplot(fig_impact)

# Visualization Section - Below the inputs and prediction
st.markdown("---")
st.markdown("## 📈 Data Visualizations")

# Create 3 columns for the visualizations
viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    # Distribution of features
    st.markdown('<div class="viz-section">', unsafe_allow_html=True)
    st.subheader("Feature Distributions")
    dist_fig, ax = plt.subplots(3, 1, figsize=(8, 8))
    
    # Feature distributions
    features = ['Avg Session Length', 'Time on App', 'Length of Membership']
    colors = ['skyblue', 'lightgreen', 'lightcoral']
    
    for i, feature in enumerate(features):
        sns.histplot(df[feature], bins=25, kde=True, color=colors[i], ax=ax[i])
        ax[i].set_title(f'Distribution of {feature}', fontsize=10)
        ax[i].set_xlabel(feature, fontsize=9)
        ax[i].tick_params(labelsize=8)
    
    plt.tight_layout()
    st.pyplot(dist_fig)
    st.markdown('</div>', unsafe_allow_html=True)

with viz_col2:
    # Correlation heatmap
    st.markdown('<div class="viz-section">', unsafe_allow_html=True)
    st.subheader("Feature Correlations")
    corr_fig, ax = plt.subplots(figsize=(8, 6))
    corr_matrix = df[['Avg Session Length', 'Time on App', 'Time on Website', 
                       'Length of Membership', 'Yearly Amount Spent']].corr()
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, annot_kws={"size": 8}, ax=ax)
    plt.tight_layout()
    st.pyplot(corr_fig)
    st.markdown('</div>', unsafe_allow_html=True)

# Full width for scatter plots
st.markdown('<div class="viz-section">', unsafe_allow_html=True)
st.subheader("Relationship with Yearly Spending")

# Create tabs for different scatter plots
tab1, tab2, tab3 = st.tabs(["Time on App vs Spending", "Membership Length vs Spending", "Session Length vs Spending"])

with tab1:
    fig_app = px.scatter(df, x="Time on App", y="Yearly Amount Spent",
                       title="Time on App vs Amount Spent",
                       labels={"Time on App": "Time on App (minutes)",
                              "Yearly Amount Spent": "Amount Spent ($)"},
                       trendline="ols")
    fig_app.update_layout(height=400)
    st.plotly_chart(fig_app, use_container_width=True)

with tab2:
    fig_membership = px.scatter(df, x="Length of Membership", y="Yearly Amount Spent",
                              title="Membership Length vs Amount Spent",
                              labels={"Length of Membership": "Membership (years)",
                                     "Yearly Amount Spent": "Amount Spent ($)"},
                              trendline="ols")
    fig_membership.update_layout(height=400)
    st.plotly_chart(fig_membership, use_container_width=True)

with tab3:
    fig_session = px.scatter(df, x="Avg Session Length", y="Yearly Amount Spent",
                           title="Session Length vs Amount Spent",
                           labels={"Avg Session Length": "Session Length (minutes)",
                                  "Yearly Amount Spent": "Amount Spent ($)"},
                           trendline="ols")
    fig_session.update_layout(height=400)
    st.plotly_chart(fig_session, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("© 2025 E-commerce Analytics | Model Version: 1.2.0")