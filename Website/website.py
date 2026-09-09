import streamlit as st
import pandas as pd
import joblib
import pickle

from visuals import create_kde_plot
from visuals import create_box_plot
from visuals import create_violin_plot
from visuals import create_heat_map
from visuals import create_bar_plot


@st.cache_resource
def get_kde_plot(data, features):
    return create_kde_plot(data, features)


@st.cache_resource
def get_box_plot(data, features):
    return create_box_plot(data, features) 

@st.cache_resource
def get_violin_plot(data, features):
    return create_violin_plot(data, features) 

@st.cache_resource
def get_heat_map(data):
    return create_heat_map(data) 

@st.cache_resource
def get_bar_plot(data, features):
    return create_bar_plot(data, features) 

# ============================================================
# 1. LOAD THE TRAINED MODEL
# ============================================================

# model_data = joblib.load("model_data.pkl")

# model = model_data["model"]
# selected_features = model_data["features"]
# X_train_5 = model_data["X_train_5"]
# y_train = model_data["y_train"]

model = pickle.load(open('Data/final_model.pkl', 'rb'))
selected_features = pickle.load(open('Data/selected_features.pkl', 'rb'))
selected_features_df = pickle.load(open('Data/selected_features_df.pkl', 'rb'))







# making the selected_features_df




# ============================================================
# 2. WEBSITE TITLE
# ============================================================

st.title("Breast Cancer Detection")

st.write(
    "Enter the five measurements below to get a prediction."
)


# ============================================================
# 3. CREATE INPUTS
# ============================================================

user_values = []

for feature in selected_features:

    value = st.number_input(
        f"Enter {feature}:",
        min_value=0.0,
    )

    user_values.append(value)


# ============================================================
# 4. PREDICT
# ============================================================

if st.button("Predict"):

    user_data = pd.DataFrame(
        [user_values],
        columns=selected_features
    )

    prediction = model.predict(user_data)

    probability = model.predict_proba(user_data)[0][1]


    # ========================================================
    # 5. DISPLAY RESULT
    # ========================================================

    if prediction[0] == 1:

        st.error("Prediction: Malignant")

    else:

        st.success("Prediction: Benign")

    st.write(
        f"Estimated malignant probability: {probability:.2%}"
    )






#show plots


st.subheader("plots")

kde_fig= get_kde_plot(selected_features_df, selected_features)
st.pyplot(kde_fig)

box_fig = get_box_plot(selected_features_df, selected_features)
st.pyplot(box_fig)

violin_fig = get_violin_plot(selected_features_df, selected_features)
st.pyplot(violin_fig)

heat_fig = get_heat_map(selected_features_df)
st.pyplot(heat_fig)

bar_fig = get_bar_plot(selected_features_df, selected_features)
st.pyplot(bar_fig)

