import os
import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import joblib
import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import joblib

HF_USERNAME = os.getenv("HF_USERNAME")
HF_TOKEN = os.getenv("HF_TOKEN")

# Set responsive layout properties and dashboard metadata
st.set_page_config(
    page_title="Visit with Us Tourism Conversion System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Fetch production serialized pipeline metrics from the hub storage layer
@st.cache_resource
def load_production_estimator_assets():
    try:
        model_artifact_path = hf_hub_download(
            repo_id=f"{HF_USERNAME}/tourism-prediction-model", 
            filename="best_tourism_model_v1.joblib",
            token=HF_TOKEN
        )
        loaded_estimator_pipeline = joblib.load(model_artifact_path)
        
        # Pull baseline features to maintain strict sequential alignment validation
        dataset_metadata_path = hf_hub_download(
            repo_id=f"{HF_USERNAME}/tourism-dataset", 
            filename="Xtrain.csv",
            repo_type="dataset"
        )
        expected_feature_sequence = pd.read_csv(dataset_metadata_path, nrows=0).columns.tolist()
        
        return loaded_estimator_pipeline, expected_feature_sequence
    except Exception as initialization_error:
        st.error(f"System Exception: Ingestion pipeline failure: {initialization_error}")
        return None, None

production_pipeline, structural_feature_order = load_production_estimator_assets()

# Inject design tokens and custom interface layouts
st.markdown("""
<style>
    /* Reset background and adjust target typography variants */
    .stApp { background-color: #fafbfc; }
    h1, h2, h3 { font-family: 'Inter', system-ui, sans-serif; font-weight: 700 !important; color: #0f172a !important; }
    
    /* Segmented card borders for user profile attributes */
    .dashboard-input-card {
        background: #ffffff;
        padding: 26px;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 24px;
    }
    
    /* Execution status result blocks */
    .outcome-panel{
        padding:22px;
        border-radius:12px;
        margin:20px 0;
    }

    .outcome-panel h3,
    .outcome-panel h4,
    .outcome-panel p{
        color:inherit !important;
    }

    .status-positive{
        background:#ecfdf5;
        border:1px solid #bbf7d0;
        color:#166534 !important;
    }

    .status-negative{
        background:#fff7ed;
        border:1px solid #fed7aa;
        color:#9a3412 !important;
    }
    
    /* Interactive command button overrides */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: #ffffff !important;
        font-weight: 600 !important;
        padding: 14px 28px !important;
        border-radius: 8px !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(29, 78, 216, 0.15);
        transition: transform 0.1s ease, box-shadow 0.1s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-0.5px);
        box-shadow: 0 6px 16px rgba(29, 78, 216, 0.25);
    }
</style>
""", unsafe_allow_html=True)

# Main container header layout
st.markdown("<p style='text-align: center; color: #2563eb; font-weight: 600; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 4px;'>Predictive Booking Engine</p>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; margin-top: 0; margin-bottom: 12px;'>✈️ Visit with Us Tourism Propensity Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #475569; font-size: 15px; max-width: 680px; margin: 0 auto 35px auto;'>Process lead demographic attributes and historical product configuration details to estimate individual package purchase conversion thresholds.</p>", unsafe_allow_html=True)

# Allocate entry parameter sections
column_demographics, column_engagement = st.columns(2, gap="large")

with column_demographics:
    # st.markdown('<div class="dashboard-input-card">', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("<h3 style='color: #2563eb !important; margin-bottom: 20px; font-size: 18px;'>📋 Consumer Demographics</h3>", unsafe_allow_html=True)
        
        input_age = st.number_input("Age (Years)", min_value=18, max_value=100, value=35, step=1)
        
        input_contact_type = st.selectbox(
            "Lead Capture Method",
            options=["Self Enquiry", "Company Invited"]
        )
        
        input_city_tier = st.selectbox(
            "Residential City Classification",
            options=[1, 2, 3],
            help="Tier 1: Major Metropolitan Area, Tier 2: Urban Center, Tier 3: Emerging Suburb"
        )
        
        input_occupation = st.selectbox(
            "Employment Classification",
            options=["Salaried", "Small Business", "Free Lancer", "Large Business"]
        )
        
        input_gender = st.selectbox("Gender Identification", options=["Male", "Female"])
        
        input_marital_status = st.selectbox(
            "Relationship Profile Status",
            options=["Single", "Married", "Divorced", "Unmarried"]
        )
        
        input_designation = st.selectbox(
            "Professional Seniority Tier",
            options=["Executive", "Manager", "Senior Manager", "AVP", "VP"]
        )
        
        input_income = st.number_input(
            "Verified Monthly Net Income (₹)",
            min_value=0.0,
            max_value=200000.0,
            value=25000.0,
            step=1000.0
        )
    # st.markdown('</div>', unsafe_allow_html=True)

with column_engagement:
    # st.markdown('<div class="dashboard-input-card">', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("<h3 style='color: #0284c7 !important; margin-bottom: 20px; font-size: 18px;'>🎯 Client Account Metrics</h3>", unsafe_allow_html=True)
        
        input_pitch_duration = st.number_input(
            "Product Consultation Length (Minutes)",
            min_value=0.0,
            max_value=60.0,
            value=15.0,
            step=0.5
        )
        
        input_total_visitors = st.number_input(
            "Total Target Adult Attendants",
            min_value=1,
            max_value=10,
            value=2,
            step=1
        )
        
        input_followup_count = st.number_input(
            "Completed Follow-up Contact Touches",
            min_value=0.0,
            max_value=10.0,
            value=3.0,
            step=1.0
        )
        
        input_product = st.selectbox(
            "Tourism Package Variant Pitched",
            options=["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
        )
        
        input_star_rating = st.selectbox(
            "Preferred Accommodations Star Tier",
            options=[3.0, 4.0, 5.0]
        )
        
        input_annual_trips = st.number_input("Number of Trips (per year)",
            min_value=0.0,
            max_value=20.0,
            value=3.0,
            step=1.0
        )
        
        input_passport_status = st.selectbox("Valid International Passport?", options=["Yes", "No"])
        
        input_satisfaction_score = st.slider(
            "Consultation Pitch Satisfaction Score",
            min_value=1,
            max_value=5,
            value=3,
            step=1
        )
        
        input_vehicle_ownership = st.selectbox("Personal Automobile Owner?", options=["Yes", "No"])
        
        input_children_count = st.number_input(
            "Total Accompanying Dependent Minor Children",
            min_value=0.0,
            max_value=5.0,
            value=0.0,
            step=1.0
        )
    # st.markdown('</div>', unsafe_allow_html=True)

# Discrete dictionary arrays to scale target categoricals
contact_mapping = {"Company Invited": 0, "Self Enquiry": 1}
occupation_mapping = {"Free Lancer": 0, "Large Business": 1, "Salaried": 2, "Small Business": 3}
gender_mapping = {"Female": 0, "Male": 1}
product_mapping = {"Basic": 0, "Deluxe": 1, "King": 2, "Standard": 3, "Super Deluxe": 4}
marital_mapping = {"Divorced": 0, "Married": 1, "Single": 2, "Unmarried": 3}
designation_mapping = {"AVP": 0, "Executive": 1, "Manager": 2, "Senior Manager": 3, "VP": 4}

# Resolve continuous categorical binary thresholds
resolved_passport = 1 if input_passport_status == "Yes" else 0
resolved_vehicle = 1 if input_vehicle_ownership == "Yes" else 0

# Encapsulate operational entry values to a structural feature frame
customer_feature_profile = pd.DataFrame([{
    'Age': input_age,
    'TypeofContact': contact_mapping[input_contact_type],
    'CityTier': input_city_tier,
    'DurationOfPitch': input_pitch_duration,
    'Occupation': occupation_mapping[input_occupation],
    'Gender': gender_mapping[input_gender],
    'NumberOfPersonVisiting': input_total_visitors,
    'NumberOfFollowups': input_followup_count,
    'ProductPitched': product_mapping[input_product],
    'PreferredPropertyStar': input_star_rating,
    'MaritalStatus': marital_mapping[input_marital_status],
    'NumberOfTrips': input_annual_trips,
    'Passport': resolved_passport,
    'PitchSatisfactionScore': input_satisfaction_score,
    'OwnCar': resolved_vehicle,
    'NumberOfChildrenVisiting': input_children_count,
    'Designation': designation_mapping[input_designation],
    'MonthlyIncome': input_income
}])

# Enforce uniform column tracking sequences based on remote pipeline properties
if structural_feature_order is not None:
    customer_feature_profile = customer_feature_profile.reindex(columns=structural_feature_order)

# Trigger estimation analytics
st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
# Trigger estimation analytics
st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

if st.button("🔮 Calculate Propensity Conversion Prediction", use_container_width=True):
    if production_pipeline is not None and structural_feature_order is not None:
        try:
            # Execute estimator array mappings
            target_conversion_class = production_pipeline.predict(customer_feature_profile)
            computed_probability_array = production_pipeline.predict_proba(customer_feature_profile)

            st.markdown(
                "<h3 style='margin-top: 25px; font-size: 20px;'>📊 Evaluation Summary</h3>",
                unsafe_allow_html=True
            )

            if target_conversion_class[0] == 1:
                st.markdown(
                    f"""
                    <div class="outcome-panel status-positive">
                        <h4>🚀 High Conversion Likelihood</h4>
                        <p>
                            The model predicts a high likelihood of conversion for this customer segment.
                        </p>
                        <h3>System Propensity Index: {computed_probability_array[0][1] * 100:.2f}%</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.success(
                    "💡 Operational Directive: Forward lead profile directly to premium booking pipelines for priority engagement."
                )

            else:
                st.markdown(
                    f"""
                    <div class="outcome-panel status-negative">
                        <h4>🛑 Standard Acquisition Profile</h4>
                        <p>
                            The model predicts this customer segment is unlikely to execute a premium purchase.
                        </p>
                        <h3>System Propensity Index: {computed_probability_array[0][1] * 100:.2f}%</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.info(
                    "💡 Alternative Campaign: Route lead to secondary transactional nurture funnels or cross-sell lower package variations."
                )

            # Display score matrix columns
            col_negative_metric, col_positive_metric = st.columns(2)

            with col_negative_metric:
                st.metric(
                    label="Calculated Non-Conversion Probability",
                    value=f"{computed_probability_array[0][0] * 100:.2f}%"
                )

            with col_positive_metric:
                st.metric(
                    label="Calculated Target Purchase Propensity",
                    value=f"{computed_probability_array[0][1] * 100:.2f}%"
                )

        except Exception as prediction_runtime_error:
            st.error(
                f"Inference Runtime Error: Evaluation halted: {prediction_runtime_error}"
            )

    else:
        st.error(
            "System Assets Unresolved: Remote repository mapping reference is missing."
        )

# Interface footer signature
st.markdown("")
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center;">
        <strong>🏢 Visit with Us Tourism Intelligence & Lead Optimization Network</strong><br>
        Cloud Deployment powered by Streamlit, Scikit-Learn pipelines, and XGBoost Ensembles
    </div>
    """,
    unsafe_allow_html=True,
)