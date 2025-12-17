import streamlit as st
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(page_title="Retail Customer Segmentation", layout="wide")
st.title("🛍️ Retail Customer Segmentation — Explainable Dashboard")

# ======================================================
# LOAD MODELS
# ======================================================
kmeans = joblib.load("kmeans.pkl")
svm_model = joblib.load("svm_ova.pkl")
svm_scaler = joblib.load("svm_scaler.pkl")
var_selector = joblib.load("variance_selector.pkl")

# ======================================================
# LOAD DATA
# ======================================================
uploaded = st.file_uploader("Upload CSV (raw or segmented)", type="csv")
df = pd.read_csv(uploaded) if uploaded else pd.read_csv("data.csv")

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# ======================================================
# CHECK IF DATA IS ALREADY SEGMENTED
# ======================================================
already_segmented = "Cluster" in df.columns

# ======================================================
# PIPELINE (ONLY IF RAW DATA)
# ======================================================
if not already_segmented:

    df = df.dropna()
    df = df.drop_duplicates()
    df = df.drop(columns=["customer_id", "signup_date"], errors="ignore")

    # Campaign aggregation
    campaign_cols = [
        'accepted_campaign_1',
        'accepted_campaign_2',
        'accepted_campaign_3',
        'accepted_campaign_4',
        'accepted_campaign_5',
        'accepted_last_campaign'
    ]
    df['campaigns_accepted_count'] = df[campaign_cols].sum(axis=1)
    df.drop(columns=campaign_cols, inplace=True)

    # Missing indicators
    df['education_missing'] = df['education_level'].isnull().astype(int)
    df['income_missing'] = df['annual_income'].isnull().astype(int)
    df['teen_missing'] = df['num_teenagers'].isnull().astype(int)

    # Imputation
    df['education_level'] = df['education_level'].fillna("Unknown")
    df['annual_income'] = df['annual_income'].fillna(df['annual_income'].median())
    df['num_teenagers'] = df['num_teenagers'].fillna(df['num_teenagers'].median())

    # Encoding
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col + "_encoded"] = le.fit_transform(df[col])

    df.drop(columns=["marital_status", "education_level"], inplace=True)

    # Variance threshold
    df_lowvar = pd.DataFrame(
        var_selector.transform(df),
        columns=df.columns[var_selector.get_support()]
    )

    # Correlation filter
    corr = df_lowvar.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > 0.85)]
    df_corr = df_lowvar.drop(columns=to_drop)

    # KMeans prediction (NO SCALING)
    df_corr["Cluster"] = kmeans.predict(df_corr)

else:
    df_corr = df.copy()

# ======================================================
# CLUSTER PROFILES
# ======================================================
cluster_profiles = (
    df_corr
    .groupby("Cluster")[["annual_income", "spend_wine", "spend_meat", "campaigns_accepted_count"]]
    .mean()
    .round(1)
)

cluster_names = {
    0: "💎 High-Value Loyal Customers",
    1: "🍷 Wine-Focused Mid Spenders",
    2: "🧺 Price-Sensitive Low Spenders"
}

cluster_actions = {
    0: "Offer premium bundles, VIP rewards, and loyalty programs.",
    1: "Promote wine-focused discounts and cross-sell meat products.",
    2: "Use aggressive promotions and entry-level offers."
}

# ======================================================
# CLUSTER VISUALS
# ======================================================
st.subheader("📊 Cluster Distribution")
fig, ax = plt.subplots()
sns.countplot(x="Cluster", data=df_corr, ax=ax)
st.pyplot(fig)

st.subheader("🧠 Average Cluster Behavior")
st.dataframe(cluster_profiles)

# ======================================================
# SVM PREDICTION + EXPLANATION
# ======================================================
st.subheader("🎯 Predict Customer Segment (SVM)")

with st.form("svm_form"):
    ai = st.number_input("Annual Income", value=float(df_corr["annual_income"].median()))
    sw = st.number_input("Spend on Wine", value=float(df_corr["spend_wine"].median()))
    sm = st.number_input("Spend on Meat", value=float(df_corr["spend_meat"].median()))
    ca = st.number_input("Campaigns Accepted", min_value=0.0, step=1.0, value=0.0)
    submit = st.form_submit_button("Predict Segment")

if submit:
    X_new = pd.DataFrame([[ai, sw, sm]],
                         columns=["annual_income", "spend_wine", "spend_meat"])
    X_scaled = svm_scaler.transform(X_new)
    pred = svm_model.predict(X_scaled)[0]

    segment_name = cluster_names.get(pred, f"Cluster {pred}")
    profile = cluster_profiles.loc[pred]

    st.success(f"🎯 **Predicted Segment: {segment_name}**")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📊 Segment Average")
        st.write(profile)
    with col2:
        st.markdown("### 💡 Business Recommendation")
        st.info(cluster_actions[pred])

    # ======================================================
    # YOU vs SEGMENT COMPARISON
    # ======================================================
    comparison_df = pd.DataFrame({
        "Feature": ["Annual Income", "Wine Spend", "Meat Spend", "Campaigns Accepted"],
        "You": [ai, sw, sm, ca],
        "Segment Average": [
            profile["annual_income"],
            profile["spend_wine"],
            profile["spend_meat"],
            profile["campaigns_accepted_count"]
        ]
    })

    # Difference %
    comparison_df["Difference %"] = comparison_df.apply(
        lambda r: round(((r["You"] - r["Segment Average"]) / r["Segment Average"]) * 100, 1)
        if r["Segment Average"] != 0 else "—",
        axis=1
    )

    st.subheader("📈 You vs Segment Average")
    st.dataframe(comparison_df.set_index("Feature"), use_container_width=True)

    # ======================================================
    # BAR CHART
    # ======================================================
    st.subheader("📊 Visual Comparison")
    chart_df = comparison_df.melt(
        id_vars="Feature",
        value_vars=["You", "Segment Average"],
        var_name="Type",
        value_name="Value"
    )
    st.bar_chart(chart_df, x="Feature", y="Value", color="Type")

    # ======================================================
    # WHY THIS SEGMENT
    # ======================================================
    st.subheader("🧠 Why This Segment?")
    reasons = []

    if ai < profile["annual_income"]:
        reasons.append("Lower income than segment average")
    else:
        reasons.append("Higher income than segment average")

    if sw > profile["spend_wine"]:
        reasons.append("Above-average wine spending")

    if sm < profile["spend_meat"]:
        reasons.append("Below-average meat spending")

    if ca > profile["campaigns_accepted_count"]:
        reasons.append("More responsive to campaigns than average")

    for r in reasons:
        st.write(f"• {r}")

# ======================================================
# DOWNLOAD
# ======================================================
st.subheader("⬇️ Download Data with Segments")
csv = df_corr.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, "segmented_customers.csv", "text/csv")
