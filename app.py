import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Mental Health in Tech Survey",
    page_icon="🧠",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
.main-title {
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #666;
    margin-bottom: 25px;
}

.section-title {
    font-size: 26px;
    font-weight: 600;
    margin-top: 20px;
}

[data-testid="stMetricValue"] {
    font-size: 28px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("mental_health_cleaned.csv")


df = load_data()

# --------------------------------------------------
# DATA CLEANING
# --------------------------------------------------

# Clean Gender
def clean_gender(value):

    if pd.isna(value):
        return "Unknown"

    value = str(value).strip().lower()

    if value in [
        "male",
        "m",
        "man",
        "cis male",
        "cis man"
    ]:
        return "Male"

    elif value in [
        "female",
        "f",
        "woman",
        "cis female",
        "cis woman"
    ]:
        return "Female"

    else:
        return "Other"


df["Gender_Clean"] = df["Gender"].apply(clean_gender)


# Clean Age
df["Age"] = pd.to_numeric(
    df["Age"],
    errors="coerce"
)

df.loc[
    (df["Age"] < 18) | (df["Age"] > 100),
    "Age"
] = np.nan


# Create Age Groups
bins = [17, 24, 34, 44, 54, 64, 100]

labels = [
    "18-24",
    "25-34",
    "35-44",
    "45-54",
    "55-64",
    "65+"
]

df["Age_Group"] = pd.cut(
    df["Age"],
    bins=bins,
    labels=labels
)


# Treatment Binary
df["Treatment_Binary"] = df["treatment"].map({
    "Yes": 1,
    "No": 0
})


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🧠 Mental Health in Tech Survey</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Exploratory Data Analysis & Interactive Streamlit Dashboard</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Overview",
        "🧠 Treatment Analysis",
        "🏢 Workplace Support",
        "🌍 Country Analysis",
        "📋 Data Summary"
    ]
)


# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.subheader("🔎 Filters")


gender_options = ["All"] + sorted(
    df["Gender_Clean"].dropna().unique().tolist()
)

selected_gender = st.sidebar.selectbox(
    "Gender",
    gender_options
)


country_options = ["All"] + sorted(
    df["Country"].dropna().unique().tolist()
)

selected_country = st.sidebar.selectbox(
    "Country",
    country_options
)


remote_options = ["All"] + sorted(
    df["remote_work"].dropna().unique().tolist()
)

selected_remote = st.sidebar.selectbox(
    "Remote Work",
    remote_options
)


treatment_options = ["All"] + sorted(
    df["treatment"].dropna().unique().tolist()
)

selected_treatment = st.sidebar.selectbox(
    "Treatment",
    treatment_options
)


# --------------------------------------------------
# APPLY FILTERS
# --------------------------------------------------

filtered_df = df.copy()


if selected_gender != "All":
    filtered_df = filtered_df[
        filtered_df["Gender_Clean"] == selected_gender
    ]


if selected_country != "All":
    filtered_df = filtered_df[
        filtered_df["Country"] == selected_country
    ]


if selected_remote != "All":
    filtered_df = filtered_df[
        filtered_df["remote_work"] == selected_remote
    ]


if selected_treatment != "All":
    filtered_df = filtered_df[
        filtered_df["treatment"] == selected_treatment
    ]


# --------------------------------------------------
# KPI FUNCTION
# --------------------------------------------------

def show_kpi_metrics(data):

    total = len(data)

    treatment_rate = (
        data["Treatment_Binary"].mean() * 100
        if total > 0 else 0
    )

    family_rate = (
        (data["family_history"] == "Yes").mean() * 100
        if total > 0 else 0
    )

    remote_rate = (
        (data["remote_work"] == "Yes").mean() * 100
        if total > 0 else 0
    )

    tech_rate = (
        (data["tech_company"] == "Yes").mean() * 100
        if total > 0 else 0
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Total Respondents",
        f"{total:,}"
    )

    col2.metric(
        "Treatment Rate",
        f"{treatment_rate:.1f}%"
    )

    col3.metric(
        "Family History",
        f"{family_rate:.1f}%"
    )

    col4.metric(
        "Remote Work",
        f"{remote_rate:.1f}%"
    )

    col5.metric(
        "Tech Company",
        f"{tech_rate:.1f}%"
    )


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if page == "🏠 Overview":

    st.markdown(
        '<div class="section-title">📌 Overall Overview</div>',
        unsafe_allow_html=True
    )

    show_kpi_metrics(filtered_df)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Treatment by Age Group")

        age_treatment = pd.crosstab(
            filtered_df["Age_Group"],
            filtered_df["treatment"]
        )

        st.bar_chart(age_treatment)

    with col2:

        st.subheader("Treatment by Gender")

        gender_treatment = pd.crosstab(
            filtered_df["Gender_Clean"],
            filtered_df["treatment"]
        )

        st.bar_chart(gender_treatment)

    st.markdown("---")

    st.subheader("Family History")

    family_counts = (
        filtered_df["family_history"]
        .value_counts()
        .rename_axis("Family History")
        .to_frame("Respondents")
    )

    st.bar_chart(family_counts)


# --------------------------------------------------
# TREATMENT ANALYSIS
# --------------------------------------------------

elif page == "🧠 Treatment Analysis":

    st.markdown(
        '<div class="section-title">🧠 Treatment Analysis</div>',
        unsafe_allow_html=True
    )

    show_kpi_metrics(filtered_df)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Treatment by Gender")

        data = pd.crosstab(
            filtered_df["Gender_Clean"],
            filtered_df["treatment"]
        )

        st.bar_chart(data)

    with col2:

        st.subheader("Treatment by Age Group")

        data = pd.crosstab(
            filtered_df["Age_Group"],
            filtered_df["treatment"]
        )

        st.bar_chart(data)

    col3, col4 = st.columns(2)

    with col3:

        st.subheader("Treatment by Family History")

        data = pd.crosstab(
            filtered_df["family_history"],
            filtered_df["treatment"]
        )

        st.bar_chart(data)

    with col4:

        st.subheader("Treatment by Work Interference")

        data = pd.crosstab(
            filtered_df["work_interfere"],
            filtered_df["treatment"]
        )

        st.bar_chart(data)

    col5, col6 = st.columns(2)

    with col5:

        st.subheader("Treatment by Remote Work")

        data = pd.crosstab(
            filtered_df["remote_work"],
            filtered_df["treatment"]
        )

        st.bar_chart(data)

    with col6:

        st.subheader("Treatment by Tech Company")

        data = pd.crosstab(
            filtered_df["tech_company"],
            filtered_df["treatment"]
        )

        st.bar_chart(data)


# --------------------------------------------------
# WORKPLACE SUPPORT
# --------------------------------------------------

elif page == "🏢 Workplace Support":

    st.markdown(
        '<div class="section-title">🏢 Workplace Mental Health Support</div>',
        unsafe_allow_html=True
    )

    show_kpi_metrics(filtered_df)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Mental Health Benefits")

        benefits = (
            filtered_df["benefits"]
            .value_counts()
            .rename_axis("Benefits")
            .to_frame("Respondents")
        )

        st.bar_chart(benefits)

    with col2:

        st.subheader("Care Options")

        care = (
            filtered_df["care_options"]
            .value_counts()
            .rename_axis("Care Options")
            .to_frame("Respondents")
        )

        st.bar_chart(care)

    col3, col4 = st.columns(2)

    with col3:

        st.subheader("Wellness Program")

        wellness = (
            filtered_df["wellness_program"]
            .value_counts()
            .rename_axis("Wellness Program")
            .to_frame("Respondents")
        )

        st.bar_chart(wellness)

    with col4:

        st.subheader("Resources to Seek Help")

        seek = (
            filtered_df["seek_help"]
            .value_counts()
            .rename_axis("Seek Help")
            .to_frame("Respondents")
        )

        st.bar_chart(seek)

    st.markdown("---")

    st.subheader("Anonymity")

    anonymity = (
        filtered_df["anonymity"]
        .value_counts()
        .rename_axis("Anonymity")
        .to_frame("Respondents")
    )

    st.bar_chart(anonymity)

    st.markdown("---")

    col5, col6 = st.columns(2)

    with col5:

        st.subheader("Benefits vs Treatment")

        data = pd.crosstab(
            filtered_df["benefits"],
            filtered_df["treatment"]
        )

        st.bar_chart(data)

    with col6:

        st.subheader("Care Options vs Treatment")

        data = pd.crosstab(
            filtered_df["care_options"],
            filtered_df["treatment"]
        )

        st.bar_chart(data)


# --------------------------------------------------
# COUNTRY ANALYSIS
# --------------------------------------------------

elif page == "🌍 Country Analysis":

    st.markdown(
        '<div class="section-title">🌍 Country Analysis</div>',
        unsafe_allow_html=True
    )

    show_kpi_metrics(filtered_df)

    st.markdown("---")

    st.subheader("Top 15 Countries by Respondent Count")

    country_counts = (
        filtered_df["Country"]
        .value_counts()
        .head(15)
    )

    st.bar_chart(country_counts)

    st.markdown("---")

    st.subheader("Treatment Rate by Country")

    country_summary = (
        filtered_df
        .groupby("Country")
        .agg(
            Respondents=("treatment", "size"),
            Treatment_Rate=(
                "Treatment_Binary",
                "mean"
            )
        )
        .reset_index()
    )

    country_summary["Treatment_Rate"] = (
        country_summary["Treatment_Rate"] * 100
    )

    country_summary = country_summary[
        country_summary["Respondents"] >= 10
    ]

    country_summary = country_summary.sort_values(
        "Treatment_Rate",
        ascending=False
    )

    st.dataframe(
        country_summary,
        use_container_width=True
    )

    if not country_summary.empty:

        st.subheader("Treatment Rate Visualization")

        chart_data = country_summary.set_index(
            "Country"
        )["Treatment_Rate"]

        st.bar_chart(chart_data)


# --------------------------------------------------
# DATA SUMMARY
# --------------------------------------------------

elif page == "📋 Data Summary":

    st.markdown(
        '<div class="section-title">📋 Dataset Summary</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Rows",
        f"{len(df):,}"
    )

    col2.metric(
        "Columns",
        f"{len(df.columns):,}"
    )

    col3.metric(
        "Duplicates",
        f"{df.duplicated().sum():,}"
    )

    col4.metric(
        "Countries",
        f"{df['Country'].nunique():,}"
    )

    st.markdown("---")

    st.subheader("Data Preview")

    st.dataframe(
        filtered_df.head(20),
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Missing Values")

    missing = (
        df.isnull()
        .sum()
        .sort_values(ascending=False)
    )

    missing = missing[missing > 0]

    if len(missing) > 0:
        missing_df = missing.rename(
            "Missing Values"
        ).to_frame()

        st.dataframe(
            missing_df,
            use_container_width=True
        )
    else:
        st.success("No missing values found.")

    st.markdown("---")

    st.subheader("Data Types")

    dtype_df = df.dtypes.astype(str).rename(
        "Data Type"
    ).to_frame()

    st.dataframe(
        dtype_df,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Project Notes")

    st.write(
        "• Dataset contains employee responses related to "
        "mental health in the technology workplace."
    )

    st.write(
        "• Analysis focuses on treatment, demographics, "
        "workplace support, and employee perceptions."
    )

    st.write(
        "• Relationships shown in this dashboard are "
        "associations and should not be interpreted as causation."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Mental Health in Tech Survey | "
    "Exploratory Data Analysis & Streamlit Dashboard"
)

