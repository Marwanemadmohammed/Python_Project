import streamlit as st
import pandas as pd
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

# st.title("Breast Cancer Detection")

st.title("🎗️ Pink-Predict: Breast Cancer Awareness & Diagnostic Dashboard")

st.markdown("""
### What is Breast Cancer?
The human body is made up of billions of cells that grow, divide, and renew in an orderly way. **Breast cancer** occurs when breast cells begin to grow out of control, dividing faster than healthy cells and forming a cluster called a **tumor**.

When doctors examine these cells under a microscope or through a biopsy, tumors fall into one of two main categories:
*   **🟢 Benign (Non-Cancerous):** These tumors are abnormal growths, but they do **not** spread into surrounding tissue or travel to other parts of the body. They are generally not life-threatening.
*   **🔴 Malignant (Cancerous):** These cells are aggressive. They can invade nearby healthy tissue and may spread to other organs if not detected and treated early.

---
### What Does This Website Do?
Medical biopsy reports contain dozens of complex cellular measurements—such as cell radius, perimeter, and surface indentations. Understanding what these numbers mean can be overwhelming.

This platform helps you:
1. **Learn the Data:** Explore how cell characteristics change when malignancy develops.
2. **Assess Risk:** Enter values from your clinical report to see your calculated probability of malignancy based on historical data.

""")

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

st.subheader("1. Density Distribution (KDE Plots)")

st.markdown("""
* Each curve represents how frequently a specific measurement occurs among patients in our clinical dataset.
* The **Blue curve** represents **Benign** cases, and the **Orange curve** represents **Malignant** cases.
* The **dotted vertical lines and numbers** point directly to the **peak (most common value)** for each group.

**Key Takeaway for Patients:**
* Notice that for every single feature, the **blue peak is far to the left** (lower values), while the **orange peak is shifted to the right** (higher values).
* For example, the typical `concave points_worst` (indentations on cell margins) for healthy cells peaks at **0.073**, whereas in cancerous tissue it surges to **0.174**. 
* If a cell's measurements fall well to the right of the orange dotted line, the likelihood of malignancy increases significantly.
""")

box_fig = get_box_plot(selected_features_df, selected_features)
st.pyplot(box_fig)

st.subheader("2. Spread and Range Analysis (Box Plots)")

st.markdown("""
* The **solid colored rectangle** covers the middle 50% of all patient cases (the typical range).
* The **line inside the box** (and the green marker) represents the **median (middle patient)**.
* The "whiskers" (lines extending outward) show the minimum and maximum boundaries, while isolated circular dots represent unusual outliers.

**Key Takeaway for Patients:**
* Look at the vertical gap between the **Blue Box (Benign)** and the **Orange Box (Malignant)**. 
* There is very little overlap between the boxes. This proves that malignant cells are consistently **larger, rougher, and more irregular in perimeter** across the majority of patients, making these metrics reliable diagnostic boundaries.
""")

violin_fig = get_violin_plot(selected_features_df, selected_features)
st.pyplot(violin_fig)

st.subheader("3. Shape and Density Profile (Violin Plots)")

st.markdown("""
* Violin plots combine density curves and box plots. The **wider** the violin shape at any given number, the more patients have that exact measurement.
* The dashed lines inside mark the 25th, 50th (median), and 75th percentiles.

**Key Takeaway for Patients:**
* The **Blue violins (Benign)** are tightly grouped and form a sharp, narrow bulge at low values. This means healthy cells are very uniform and predictable in size.
* The **Orange violins (Malignant)** are elongated and widely stretched out. This illustrates that cancerous cells grow unpredictably, varying greatly in size and shape from one patient to another.
""")

heat_fig = get_heat_map(selected_features_df)
st.pyplot(heat_fig)



bar_fig = get_bar_plot(selected_features_df, selected_features)
st.pyplot(bar_fig)

st.subheader("4. Feature Importance & Diagnostic Correlation")

st.markdown("""
* **Correlation** measures how strongly a cellular measurement is linked to whether a tumor is malignant. Scores range from **0.0 (no connection)** to **1.0 (perfect connection)**.
* Out of 30 different measurements recorded in clinical laboratory reports, our analysis isolated the **Top 5 highest indicators**:
  1. `concave points_worst` (0.79)
  2. `concave points_mean` (0.78)
  3. `perimeter_worst` (0.77)
  4. `radius_worst` (0.77)
  5. `perimeter_mean` (0.73)

**Key Takeaway for Patients:**
* A correlation score above **0.70** is considered statistically **very strong**. 
* Cell indentations (`concave points`) and overall cell boundary dimensions (`perimeter` and `radius`) are the most decisive structural flags of breast malignancy. This is why our diagnostic form requires only these 5 inputs rather than all 30.
""")

