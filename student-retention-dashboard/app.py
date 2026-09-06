```python
import os
import time
import pandas as pd
import requests
import streamlit as st

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Retention Strategy",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS - UI / UX
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #f7f9fc;
    }

    /* Main content */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Header */
    .main-header {
        padding: 1.5rem 0 1rem 0;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #172033;
    }

    .main-subtitle {
        font-size: 1.05rem;
        color: #667085;
        margin-bottom: 1.5rem;
    }

    /* Cards */
    .metric-card {
        background: white;
        padding: 1.3rem;
        border-radius: 14px;
        border: 1px solid #e6eaf0;
        box-shadow: 0 2px 8px rgba(16, 24, 40, 0.05);
        min-height: 125px;
    }

    .metric-title {
        color: #667085;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        color: #172033;
        font-size: 1.8rem;
        font-weight: 700;
    }

    /* Section cards */
    .section-card {
        background: white;
        padding: 1.5rem;
        border-radius: 14px;
        border: 1px solid #e6eaf0;
        box-shadow: 0 2px 8px rgba(16, 24, 40, 0.04);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 650;
        color: #172033;
        margin-bottom: 0.7rem;
    }

    .section-description {
        color: #667085;
        font-size: 0.92rem;
        margin-bottom: 1rem;
    }

    /* Department badge */
    .department-badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        background-color: #eef4ff;
        color: #3157c7;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        background: white;
        padding: 1rem;
        border-radius: 14px;
        border: 1px solid #e6eaf0;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        padding: 0.65rem 1rem;
        font-weight: 600;
        font-size: 1rem;
    }

    /* Info box */
    .info-card {
        background: #eef4ff;
        border-left: 4px solid #4c6fff;
        padding: 1rem;
        border-radius: 8px;
        color: #344054;
        margin: 1rem 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #98a2b3;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e6eaf0;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# API KEYS
# ============================================================

try:
    os.environ["CREWAI_BEARER_TOKEN"] = st.secrets["CREWAI_BEARER_TOKEN"]
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    st.error("⚠️ API keys are not configured correctly in Streamlit Secrets.")
    st.stop()


# ============================================================
# CREWAI CONFIGURATION
# ============================================================

BASE_URL = "https://enrollment-retention-management-crew-v1-c21-82c3d780.crewai.com"

KICKOFF_URL = f"{BASE_URL}/kickoff"
STATUS_URL = f"{BASE_URL}/status"

HEADERS = {
    "Authorization": f"Bearer {os.environ['CREWAI_BEARER_TOKEN']}",
    "Content-Type": "application/json"
}


# ============================================================
# GEMINI LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)


# ============================================================
# DEPARTMENT PROMPT
# ============================================================

dept_template = """
You are an academic management strategist.

Analyze the following department-level student performance data.

Department: {Department}
Average GPA: {Avg_GPA}
Average Attendance: {Avg_Attendance}%
Average Engagement Score: {Avg_Engagement} out of 3

Provide a concise administration-focused analysis.

Structure your response exactly under these headings:

### Key Strengths
Mention the positive aspects of the department.

### Key Risks
Identify potential academic, attendance, engagement, retention, or enrollment concerns.

### Recommended Actions
Give 3 to 5 practical actions administrators can take.

### Overall Assessment
Give a short concluding assessment of the department.

Keep the response professional, clear, and actionable.
"""

dept_prompt = PromptTemplate(
    template=dept_template,
    input_variables=[
        "Department",
        "Avg_GPA",
        "Avg_Attendance",
        "Avg_Engagement"
    ]
)

dept_chain = LLMChain(
    llm=llm,
    prompt=dept_prompt
)


# ============================================================
# CREWAI FUNCTIONS
# ============================================================

def start_crew(department_name):

    data = {
        "inputs": {
            "department_name": department_name
        }
    }

    response = requests.post(
        KICKOFF_URL,
        json=data,
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()

    return response.json()["kickoff_id"]


def check_status(kickoff_id):

    max_attempts = 60

    for _ in range(max_attempts):

        response = requests.get(
            f"{STATUS_URL}/{kickoff_id}",
            headers=HEADERS,
            timeout=30
        )

        response.raise_for_status()

        result = response.json()

        if result.get("state") == "SUCCESS":
            return result.get(
                "result",
                "No output returned from CrewAI."
            )

        elif result.get("state") == "FAILED":
            return "CrewAI execution failed."

        time.sleep(2)

    return "CrewAI analysis timed out. Please try again."


# ============================================================
# PAGE HEADER
# ============================================================

st.markdown("""
<div class="main-header">

<div class="main-title">
🎓 Student Enrollment & Retention Strategy
</div>

<div class="main-subtitle">
Transform department-level student data into actionable academic,
retention, and enrollment insights using Gemini + CrewAI.
</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🎯 Analysis Center")

    st.markdown("""
    Upload your department CSV and select a department to generate
    an AI-powered performance and retention strategy.
    """)

    st.divider()

    st.markdown("### 📌 Workflow")

    st.markdown("""
    **1.** Upload CSV  
    **2.** Select department  
    **3.** Review performance  
    **4.** Generate AI insights  
    **5.** Review retention strategy
    """)

    st.divider()

    st.caption("Powered by Gemini + LangChain + CrewAI")


# ============================================================
# FILE UPLOAD
# ============================================================

st.markdown("### 📂 Upload Department Data")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing student department data",
    type=["csv"],
    help="CSV should contain Department, GPA, Attendance and Engagement columns."
)


if uploaded_file is None:

    st.markdown("""
    <div class="info-card">

    <strong>👋 Welcome!</strong><br><br>

    Upload your department CSV file to begin the analysis.

    The dashboard will calculate department-level performance
    indicators and generate AI-powered recommendations.

    </div>
    """, unsafe_allow_html=True)

    st.stop()


# ============================================================
# READ CSV
# ============================================================

try:

    df = pd.read_csv(uploaded_file)

except Exception as e:

    st.error(f"❌ Unable to read the CSV file: {e}")
    st.stop()


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Department",
    "GPA",
    "Attendance",
    "Engagement"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:

    st.error(
        "❌ Missing required columns: "
        + ", ".join(missing_columns)
    )

    st.info(
        "Required columns: Department, GPA, Attendance, Engagement"
    )

    st.stop()


# ============================================================
# DATA PREPARATION
# ============================================================

engagement_map = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}

df["Engagement_numeric"] = (
    df["Engagement"]
    .astype(str)
    .str.strip()
    .str.title()
    .map(engagement_map)
    .fillna(0)
)


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.markdown("### 📋 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Students",
        len(df)
    )

with col2:
    st.metric(
        "Departments",
        df["Department"].nunique()
    )

with col3:
    st.metric(
        "Average GPA",
        f"{df['GPA'].mean():.2f}"
    )

with col4:
    st.metric(
        "Average Attendance",
        f"{df['Attendance'].mean():.1f}%"
    )


with st.expander("🔎 View Uploaded Data"):

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DEPARTMENT SELECTION
# ============================================================

st.markdown("### 🏢 Department Analysis")

department_name = st.selectbox(
    "Select a department",
    sorted(df["Department"].dropna().unique()),
    help="Choose the department you want to analyze."
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze = st.button(
    "🚀 Analyze Department",
    type="primary",
    use_container_width=True
)


if analyze:

    # --------------------------------------------------------
    # FILTER DEPARTMENT
    # --------------------------------------------------------

    dept_df = df[
        df["Department"] == department_name
    ]

    avg_gpa = round(
        dept_df["GPA"].mean(),
        2
    )

    avg_attendance = round(
        dept_df["Attendance"].mean(),
        2
    )

    avg_engagement = round(
        dept_df["Engagement_numeric"].mean(),
        2
    )

    student_count = len(dept_df)


    # --------------------------------------------------------
    # DEPARTMENT HEADER
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="section-card">

        <div class="department-badge">
        🏢 {department_name}
        </div>

        <div class="section-title">
        Department Performance Overview
        </div>

        <div class="section-description">
        Analysis based on {student_count} student records.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # METRIC CARDS
    # --------------------------------------------------------

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-title">
            📚 Average GPA
            </div>

            <div class="metric-value">
            {avg_gpa}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with metric2:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-title">
            📅 Attendance
            </div>

            <div class="metric-value">
            {avg_attendance}%
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with metric3:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-title">
            📈 Engagement
            </div>

            <div class="metric-value">
            {avg_engagement}/3
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with metric4:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-title">
            👥 Students
            </div>

            <div class="metric-value">
            {student_count}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.write("")


    # ========================================================
    # GEMINI ANALYSIS
    # ========================================================

    st.markdown("""
    <div class="section-card">

    <div class="section-title">
    🧠 AI Department Insights
    </div>

    <div class="section-description">
    Gemini analyzes the department's academic performance,
    attendance, and engagement indicators.
    </div>

    </div>
    """, unsafe_allow_html=True)


    try:

        with st.spinner(
            "🧠 Gemini is analyzing department performance..."
        ):

            dept_result = dept_chain.invoke({
                "Department": department_name,
                "Avg_GPA": avg_gpa,
                "Avg_Attendance": avg_attendance,
                "Avg_Engagement": avg_engagement
            })

            dept_summary = dept_result["text"]


        st.markdown(
            '<div class="section-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            dept_summary
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    except Exception as e:

        st.error(
            f"❌ Gemini analysis failed: {e}"
        )

        st.stop()


    # ========================================================
    # CREWAI RETENTION STRATEGY
    # ========================================================

    st.markdown("""
    <div class="section-card">

    <div class="section-title">
    🤖 AI Retention & Enrollment Strategy
    </div>

    <div class="section-description">
    CrewAI generates department-specific strategies
    for improving student retention and enrollment.
    </div>

    </div>
    """, unsafe_allow_html=True)


    try:

        with st.spinner(
            "🤖 CrewAI is preparing the retention strategy..."
        ):

            kickoff_id = start_crew(
                department_name
            )

            analysis = check_status(
                kickoff_id
            )


        st.markdown(
            '<div class="section-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            analysis
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    except Exception as e:

        st.error(
            f"❌ CrewAI request failed: {e}"
        )


    # ========================================================
    # SUCCESS MESSAGE
    # ========================================================

    st.success(
        f"✅ Analysis completed successfully for {department_name}."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

Student Enrollment & Retention Strategy Dashboard  
<br>
Gemini • LangChain • CrewAI • Streamlit

</div>
""", unsafe_allow_html=True)
```

### One more change: `requirements.txt`

Because you switched from Groq to Gemini, make sure your GitHub `requirements.txt` contains the Gemini integration.

I recommend:

```text
streamlit==1.38.0
pandas==2.2.3
requests==2.32.3
langchain==0.2.16
langchain-google-genai==1.0.10
```

You can remove:

```text
langchain-groq==0.1.6
```

### ⚠️ Before deploying

There is one thing I'd check before you commit this: **`gemini-2.5-flash` availability with your particular Google API key/free tier**. If the model isn't available for that key, you'll get a model-access error even though the UI is correct.

Also, your CrewAI endpoint may return Markdown/plain text. The new UI uses `st.markdown()` so headings, bullets, and formatting will look much cleaner than `st.write()`.

The resulting flow will look roughly like:

**🎓 Dashboard**

→ Upload CSV

→ **Dataset Overview**
`Students | Departments | Avg GPA | Avg Attendance`

→ **🏢 Select Department**

→ **Performance Cards**
`GPA | Attendance | Engagement | Students`

→ **🧠 AI Department Insights**
Strengths → Risks → Recommended Actions → Overall Assessment

→ **🤖 AI Retention & Enrollment Strategy**

→ **✅ Analysis completed**

This should give the project a much more polished **dashboard/product feel** rather than looking like a basic Streamlit prototype.
