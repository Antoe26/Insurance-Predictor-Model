import streamlit as st
import pandas as pd
import joblib
from utils import convert_age_to_midpoint, convert_income_to_midpoint

# 1. Load the pipeline (Ensuring version alignment)
model = joblib.load('insurance_model_pipeline.pkl') 

# 2. Setup Page Config
st.set_page_config(page_title="Insurance Predictor", layout="wide")

# 3. Sidebar Inputs
st.sidebar.header("Respondent Information")
st.sidebar.write("Adjust the details to run a prediction.")

# Collecting inputs in the sidebar to keep the main page clean
age_range = st.sidebar.selectbox("Age Range", 
                                 options=["18-30", "31-40", "41-50", "51-60", "60+"], index=None,
                                 placeholder="Select Your Age Range...")
gender = st.sidebar.selectbox("Gender", 
                              options=["male", "female" ],
                              index=None,
                              placeholder="Select Gender...")
marital = st.sidebar.selectbox("Marital Status",
                                options=["single", "married", "divorced", "widowed"],
                                index=None,
                                placeholder="Select Marital Status...")
children = st.sidebar.number_input("Number of Children", min_value=0, max_value=20, value=0)
employment = st.sidebar.selectbox("Employment Status", 
                                  options=["unemployed", "self-employed", "employed", "student"],
                                  index=None,
                                  placeholder="Select Employment Status...")
income_range = st.sidebar.selectbox("Monthly Household Income", 
                    options=["less than 10000", "10001-20000", "20001-30000", "30001-40000", "40001-50000", "50001-60000", "60+"],
                    index=None,
                    placeholder="Select Household Income...")
location = st.sidebar.text_input("Location (Coordinates)", "-0.2742 36.0583")

#  Main Page Display
# Adding healthcare banner
st.image("https://images.unsplash.com/photo-1516549655169-df83a0774514?auto=format&fit=crop&q=80&w=1000", 
         use_container_width=True)

st.title("🛡️ Insurance Coverage Predictor")
st.markdown("""
This tool uses a **Random Forest Machine Learning model** to analyze demographic data 
and predict if a respondent is likely to have health insurance coverage.
""")

st.divider()

# Insights showing factors affecting prediction
if st.sidebar.checkbox("Show Model Insights"):
    st.subheader("📊 What influences the prediction? Factors affecting Prediction.")
    
    try:
        # 1. Get the model from the pipeline
        rf_model = model.named_steps['model']
        importances = rf_model.feature_importances_
        
        # 2. Get feature names more reliably
        # We try to get them from the Processor step
        processor = model.named_steps['processor']
        
        try:
            # Standard way for newer scikit-learn
            feature_names = processor.get_feature_names_out()
        except:
            # Fallback for older versions or specific configurations
            num_cols = processor.transformers_[0][2]
            cat_transformer = processor.transformers_[1][1]
            cat_cols = cat_transformer.named_steps['onehot'].get_feature_names_out(processor.transformers_[1][2])
            feature_names = list(num_cols) + list(cat_cols)

        # 3. Create the Series and plot
        feat_importances = pd.Series(importances, index=feature_names)
        
        # Clean up the names (removes 'cat__' or 'num__' prefixes for a prettier chart)
        feat_importances.index = [name.split('__')[-1] for name in feat_importances.index]
        
        top_10 = feat_importances.nlargest(10)
        st.bar_chart(top_10)
        st.info("The chart shows the top 10 factors affecting the prediction.")

    except Exception as e:
        # This will print the actual error to your Streamlit app so you can see it
        st.warning(f"Feature importance alignment error: {e}")


#  Prediction 
if st.sidebar.button("Run Analysis"):
    # Create the dataframe matching the training features exactly 
    
    if age_range is None or gender is None or marital is None or employment is None or income_range is None:
        st.error("Please select all required fields before running the Analysis!!")
    else:
        input_data = pd.DataFrame({  
        'Age': [age_range],
        'Gender': [gender],
        'Marital Status': [marital],
        'Children': [float(children)],
        'Employment Status': [employment],
        'Monthly Household Income': [income_range],
        'Location': [location]
        })

    # 2. Convert ranges to midpoints (using your functions from utils.py)
        input_data['Age'] = input_data['Age'].apply(convert_age_to_midpoint)
        input_data['Monthly Household Income'] = input_data['Monthly Household Income'].apply(convert_income_to_midpoint)


    # Get prediction from the loaded model 
        prediction = model.predict(input_data)

    # Display results visually
        st.subheader("Analysis Result")
        if prediction[0].lower() == 'yes':
            st.success("### ✅ Likely Covered")
            st.balloons() 
            st.write("Based on the provided data, this individual likely HAS insurance coverage.")
        else:
            st.error("### ❌ Likely Not Covered")
            st.write("Based on the provided data, this individual likely DOES NOT have insurance coverage.") 
            
else:
   st.info("👈 Please enter the respondent's details in the sidebar and click **'Run Analysis'**.")

