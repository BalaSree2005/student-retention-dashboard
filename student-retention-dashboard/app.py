import os
import time
import re
from io import BytesIO
from html import escape

import pandas as pd
import requests
import streamlit as st

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)


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
# CUSTOM CSS
# ============================================================

st.html("""
<style>

.stApp {
    background-color: #f6f8fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

.main-header {
    padding: 1rem 0 1.5rem 0;
}

.main-title {
    font-size: 2.4rem;
    font-weight: 750;
    color: #172033;
    margin-bottom: 0.4rem;
}

.main-subtitle {
    font-size: 1.05rem;
    color: #667085;
    max-width: 850px;
}

.section-heading {
    font-size: 1.45rem;
    font-weight: 700;
    color: #172033;
    margin-top: 1.8rem;
    margin-bottom: 0.8rem;
}

.section-description {
    color: #667085;
    font-size: 0.92rem;
    margin-bottom: 1rem;
}

.metric-card {
    background: white;
    padding: 1.25rem;
    border-radius: 14px;
    border: 1px solid #e5e9f0;
    box-shadow: 0 3px 10px rgba(16, 24, 40, 0.05);
    min-height: 115px;
}

.metric-title {
    color: #667085;
    font-size: 0.88rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

.metric-value {
    color: #172033;
    font-size: 1.8rem;
    font-weight: 750;
}

.department-header {
    background: white;
    padding: 1.4rem 1.6rem;
    border-radius: 14px;
    border: 1px solid #e5e9f0;
    box-shadow: 0 3px 10px rgba(16, 24, 40, 0.04);
    margin-top: 1.2rem;
}

.department-badge {
    display: inline-block;
    background: #eef4ff;
    color: #3157c7;
    padding: 0.35rem 0.8rem;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 700;
    margin-bottom: 0.7rem;
}

.department-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #172033;
}

.department-description {
    color: #667085;
    font-size: 0.9rem;
}

.strength-card {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-left: 5px solid #22c55e;
    padding: 1.25rem 1.4rem;
    border-radius: 12px;
    margin-bottom: 1rem;
}

.strength-title {
    color: #166534;
    font-weight: 750;
    font-size: 1.05rem;
    margin-bottom: 0.5rem;
}

.risk-card {
    background: #fffaf5;
    border: 1px solid #fed7aa;
    border-left: 5px solid #f97316;
    padding: 1.2rem 1.35rem;
    border-radius: 12px;
    margin-bottom: 0.8rem;
}

.risk-title {
    color: #9a3412;
    font-weight: 700;
    font-size: 1rem;
    margin-bottom: 0.4rem;
}

.action-card {
    background: white;
    border: 1px solid #dbe3f0;
    border-left: 5px solid #4c6fff;
    padding: 1.2rem 1.35rem;
    border-radius: 12px;
    margin-bottom: 0.8rem;
}

.action-number {
    color: #3157c7;
    font-weight: 750;
    font-size: 0.9rem;
}

.action-title {
    color: #172033;
    font-size: 1rem;
    font-weight: 700;
}

.action-description {
    color: #475467;
    font-size: 0.9rem;
    line-height: 1.55;
}

.insight-text {
    color: #475467;
    font-size: 0.9rem;
    line-height: 1.55;
}

.assessment-card {
    background: #eef4ff;
    border: 1px solid #c7d7fe;
    border-left: 5px solid #4c6fff;
    padding: 1.3rem 1.4rem;
    border-radius: 12px;
}

.assessment-title {
    color: #243b8f;
    font-size: 1.05rem;
    font-weight: 750;
}

.assessment-text {
    color: #344054;
    font-size: 0.92rem;
    line-height: 1.6;
}

.strategy-card {
    background: white;
    border: 1px solid #e5e9f0;
    border-radius: 14px;
    padding: 1.35rem 1.45rem;
    margin-bottom: 1rem;
}

.strategy-number {
    color: #3157c7;
    font-size: 0.85rem;
    font-weight: 750;
}

.strategy-title {
    color: #172033;
    font-size: 1.1rem;
    font-weight: 750;
}

.strategy-subtitle {
    color: #344054;
    font-size: 0.98rem;
    font-weight: 700;
    margin-top: 0.7rem;
}

.strategy-text {
    color: #475467;
    font-size: 0.9rem;
    line-height: 1.6;
    margin-top: 0.25rem;
}

.download-card {
    background: white;
    border: 1px solid #dbe3f0;
    border-radius: 14px;
    padding: 1.5rem;
    margin-top: 2rem;
    text-align: center;
}

[data-testid="stFileUploader"] {
    background: white;
    padding: 1rem;
    border-radius: 14px;
    border: 1px solid #e5e9f0;
}

.stButton > button {
    border-radius: 10px;
    min-height: 45px;
    font-weight: 650;
}

.footer {
    text-align: center;
    color: #98a2b3;
    font-size: 0.8rem;
    margin-top: 3rem;
    padding-top: 1.2rem;
    border-top: 1px solid #e5e9f0;
}

</style>
""")


# ============================================================
# API KEYS
# ============================================================

try:

    os.environ["CREWAI_BEARER_TOKEN"] = (
        st.secrets["CREWAI_BEARER_TOKEN"]
    )

    os.environ["GOOGLE_API_KEY"] = (
        st.secrets["GOOGLE_API_KEY"]
    )

except Exception:

    st.error(
        "⚠️ API keys are not configured correctly in "
        "Streamlit Secrets."
    )

    st.stop()


# ============================================================
# CREWAI CONFIGURATION
# ============================================================

BASE_URL = (
    "https://enrollment-retention-management-crew-v1-c21-82c3d780.crewai.com"
)

KICKOFF_URL = f"{BASE_URL}/kickoff"
STATUS_URL = f"{BASE_URL}/status"

HEADERS = {
    "Authorization": (
        f"Bearer {os.environ['CREWAI_BEARER_TOKEN']}"
    ),
    "Content-Type": "application/json"
}


# ============================================================
# GEMINI
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)


# ============================================================
# GEMINI PROMPT
# ============================================================

dept_template = """
You are an academic management strategist.

Analyze the following department-level student performance data.

Department: {Department}
Average GPA: {Avg_GPA}
Average Attendance: {Avg_Attendance}%
Average Engagement Score: {Avg_Engagement} out of 3

Provide a concise administration-focused analysis.

Structure your response exactly using these headings:

### Key Strengths
Give 2 to 3 concise positive observations.

### Key Risks
Give 3 to 5 concise risks. Each risk should be one clear point.

### Recommended Actions
Give 3 to 5 practical actions. Each action should have a short title followed by a concise explanation.

### Overall Assessment
Give one concise concluding paragraph.

Keep the response professional, clear and actionable.
Avoid unnecessary introduction or conclusion.
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

    try:

        response = requests.post(
            KICKOFF_URL,
            json=data,
            headers=HEADERS,
            timeout=120
        )

        response.raise_for_status()

        result = response.json()

        if "kickoff_id" not in result:

            raise Exception(
                f"CrewAI response did not contain kickoff_id: "
                f"{result}"
            )

        return result["kickoff_id"]

    except requests.exceptions.Timeout:

        raise Exception(
            "CrewAI server took more than 120 seconds "
            "to respond to the kickoff request."
        )

    except requests.exceptions.ConnectionError:

        raise Exception(
            "Could not connect to the CrewAI server."
        )

    except requests.exceptions.HTTPError as e:

        raise Exception(
            f"CrewAI HTTP error: {e}"
        )

    except Exception as e:

        raise Exception(
            f"CrewAI kickoff failed: {e}"
        )


def check_status(kickoff_id):

    max_attempts = 90

    for _ in range(max_attempts):

        try:

            response = requests.get(
                f"{STATUS_URL}/{kickoff_id}",
                headers=HEADERS,
                timeout=60
            )

            response.raise_for_status()

            result = response.json()

            state = result.get("state")

            if state == "SUCCESS":

                return result.get(
                    "result",
                    "No output returned from CrewAI."
                )

            elif state == "FAILED":

                return (
                    "CrewAI execution failed."
                )

            time.sleep(2)

        except requests.exceptions.Timeout:

            # Continue checking instead of immediately failing
            continue

        except Exception as e:

            return (
                f"CrewAI status check failed: {e}"
            )

    return (
        "CrewAI analysis timed out after "
        "approximately 3 minutes."
    )

# ============================================================
# TEXT CLEANING HELPERS
# ============================================================

def clean_text(text):

    """Remove markdown formatting for cleaner display."""

    text = str(text)

    text = re.sub(
        r"\*\*(.*?)\*\*",
        r"\1",
        text
    )

    text = re.sub(
        r"\*(.*?)\*",
        r"\1",
        text
    )

    text = re.sub(
        r"__(.*?)__",
        r"\1",
        text
    )

    text = re.sub(
        r"`(.*?)`",
        r"\1",
        text
    )

    return text.strip()


def split_gemini_sections(text):

    sections = {
        "Key Strengths": "",
        "Key Risks": "",
        "Recommended Actions": "",
        "Overall Assessment": ""
    }

    current = None

    for line in str(text).splitlines():

        line = line.strip()

        if not line:
            continue

        normalized = re.sub(
            r"^#+\s*",
            "",
            line
        ).strip()

        matched = False

        for heading in sections:

            if normalized.lower() == heading.lower():

                current = heading
                matched = True
                break

        if matched:
            continue

        if current:
            sections[current] += line + "\n"

    return sections


def extract_bullets(text):

    items = []

    for line in str(text).splitlines():

        line = line.strip()

        if not line:
            continue

        if line.startswith("- "):

            items.append(
                clean_text(line[2:])
            )

        elif line.startswith("* "):

            items.append(
                clean_text(line[2:])
            )

        elif re.match(
            r"^\d+[\.\)]\s+",
            line
        ):

            item = re.sub(
                r"^\d+[\.\)]\s+",
                "",
                line
            )

            items.append(
                clean_text(item)
            )

    return items


# ============================================================
# GEMINI OUTPUT RENDERING
# ============================================================

def render_gemini_output(text):

    sections = split_gemini_sections(text)

    # --------------------------------------------------------
    # KEY STRENGTHS
    # --------------------------------------------------------

    st.html(
        '<div class="section-heading">'
        '🌟 Key Strengths'
        '</div>'
    )

    strengths = extract_bullets(
        sections["Key Strengths"]
    )

    if strengths:

        for strength in strengths:

            st.html(
                f"""
                <div class="strength-card">

                    <div class="strength-title">
                        ✓ Positive Indicator
                    </div>

                    <div class="insight-text">
                        {escape(strength)}
                    </div>

                </div>
                """
            )

    else:

        st.html(
            f"""
            <div class="strength-card">

                <div class="insight-text">
                    {escape(
                        clean_text(
                            sections["Key Strengths"]
                        )
                    )}
                </div>

            </div>
            """
        )


    # --------------------------------------------------------
    # KEY RISKS
    # --------------------------------------------------------

    st.html(
        '<div class="section-heading">'
        '⚠️ Key Risks'
        '</div>'
    )

    risks = extract_bullets(
        sections["Key Risks"]
    )

    if risks:

        for index, risk in enumerate(
            risks,
            1
        ):

            st.html(
                f"""
                <div class="risk-card">

                    <div class="risk-title">
                        Risk {index}
                    </div>

                    <div class="insight-text">
                        {escape(risk)}
                    </div>

                </div>
                """
            )

    else:

        st.html(
            f"""
            <div class="risk-card">

                <div class="insight-text">
                    {escape(
                        clean_text(
                            sections["Key Risks"]
                        )
                    )}
                </div>

            </div>
            """
        )


    # --------------------------------------------------------
    # RECOMMENDED ACTIONS
    # --------------------------------------------------------

    st.html(
        '<div class="section-heading">'
        '🚀 Recommended Actions'
        '</div>'
    )

    actions = extract_bullets(
        sections["Recommended Actions"]
    )

    if actions:

        for index, action in enumerate(
            actions,
            1
        ):

            if ":" in action:

                title, description = action.split(
                    ":",
                    1
                )

            else:

                title = (
                    f"Recommended Action {index}"
                )

                description = action

            st.html(
                f"""
                <div class="action-card">

                    <div class="action-number">
                        ACTION {index}
                    </div>

                    <div class="action-title">
                        {escape(title.strip())}
                    </div>

                    <div class="action-description">
                        {escape(description.strip())}
                    </div>

                </div>
                """
            )

    else:

        st.html(
            f"""
            <div class="action-card">

                <div class="action-description">
                    {escape(
                        clean_text(
                            sections["Recommended Actions"]
                        )
                    )}
                </div>

            </div>
            """
        )


    # --------------------------------------------------------
    # OVERALL ASSESSMENT
    # --------------------------------------------------------

    st.html(
        '<div class="section-heading">'
        '📌 Overall Assessment'
        '</div>'
    )

    assessment = clean_text(
        sections["Overall Assessment"]
    )

    st.html(
        f"""
        <div class="assessment-card">

            <div class="assessment-title">
                Department Outlook
            </div>

            <div class="assessment-text">
                {escape(assessment)}
            </div>

        </div>
        """
    )


# ============================================================
# CREWAI PARSING
# ============================================================

def parse_crewai_strategy(text):
    """
    Parse CrewAI's enrollment/retention strategy into:
    
    Major Strategy
        -> Target / Subsection
            -> Strategy
            -> Implementation
            -> Expected Outcome
    """

    strategies = []

    current_strategy = None
    current_target = None
    current_detail = None

    lines = str(text).splitlines()

    for raw_line in lines:

        line = raw_line.strip()

        if not line:
            continue

        # Remove markdown formatting
        line = re.sub(
            r"\*\*(.*?)\*\*",
            r"\1",
            line
        )

        line = re.sub(
            r"__(.*?)__",
            r"\1",
            line
        )

        line = line.strip()

        # ----------------------------------------------------
        # Ignore separators
        # ----------------------------------------------------

        if line in ["---", "***", "___"]:
            continue

        # ----------------------------------------------------
        # Remove markdown heading symbols
        # ----------------------------------------------------

        clean = re.sub(
            r"^#+\s*",
            "",
            line
        ).strip()

        # ----------------------------------------------------
        # Ignore report title
        # ----------------------------------------------------

        if clean.lower().startswith(
            "enrollment strategy report"
        ):
            continue

        if clean.lower().startswith(
            "objective:"
        ):
            continue

        # ----------------------------------------------------
        # MAJOR STRATEGY
        # Example:
        # 1. Recruitment Strategies Segmented by GPA Levels
        # ----------------------------------------------------

        major_match = re.match(
            r"^(\d+)\.\s+(.+)$",
            clean
        )

        if major_match:

            # Save previous strategy
            if current_strategy is not None:
                strategies.append(
                    current_strategy
                )

            current_strategy = {
                "number": major_match.group(1),
                "title": major_match.group(2).strip(),
                "targets": []
            }

            current_target = None
            current_detail = None

            continue

        # ----------------------------------------------------
        # TARGET / SUBSECTION
        #
        # Examples:
        # High GPA Target (3.5 and above)
        # Moderate GPA Target (2.8 - 3.4)
        # Low GPA Target (Below 2.8)
        # Engagement Strategy
        # Attendance Initiative
        # Partnership Strategy
        # Successful Segments to Target
        # At-Risk Segments to Avoid
        # ----------------------------------------------------

        target_patterns = [
            r"^High GPA Target.*",
            r"^Moderate GPA Target.*",
            r"^Low GPA Target.*",
            r"^High[- ]Engagement.*",
            r"^Engagement Strategy.*",
            r"^Attendance Initiative.*",
            r"^Partnership Strategy.*",
            r"^Successful Segments to Target.*",
            r"^At-Risk Segments to Avoid.*"
        ]

        is_target = any(
            re.match(
                pattern,
                clean,
                re.IGNORECASE
            )
            for pattern in target_patterns
        )

        if is_target and current_strategy:

            current_target = {
                "title": clean,
                "details": []
            }

            current_strategy[
                "targets"
            ].append(
                current_target
            )

            current_detail = None

            continue

        # ----------------------------------------------------
        # GENERIC SUBHEADING
        #
        # This catches headings that CrewAI may generate
        # that aren't explicitly listed above.
        # ----------------------------------------------------

        if (
            current_strategy
            and not clean.startswith("-")
            and not re.match(
                r"^(Strategy|Implementation|Expected Outcome):",
                clean,
                re.IGNORECASE
            )
            and len(clean) < 120
        ):

            # Treat short standalone text as a subsection
            if (
                current_target is None
                or (
                    current_target
                    and current_target["details"]
                    and current_detail is not None
                )
            ):

                current_target = {
                    "title": clean,
                    "details": []
                }

                current_strategy[
                    "targets"
                ].append(
                    current_target
                )

                current_detail = None

                continue

        # ----------------------------------------------------
        # DETAIL LINES
        #
        # Strategy:
        # Implementation:
        # Expected Outcome:
        # ----------------------------------------------------

        detail_match = re.match(
            r"^(Strategy|Implementation|Expected Outcome)\s*:\s*(.*)$",
            clean,
            re.IGNORECASE
        )

        if detail_match:

            label = detail_match.group(1).strip()

            content = detail_match.group(2).strip()

            # Create target if missing
            if current_target is None:

                current_target = {
                    "title": "Recommended Approach",
                    "details": []
                }

                current_strategy[
                    "targets"
                ].append(
                    current_target
                )

            detail = {
                "label": label,
                "text": content
            }

            current_target[
                "details"
            ].append(
                detail
            )

            current_detail = detail

            continue

        # ----------------------------------------------------
        # BULLET CONTINUATION
        # ----------------------------------------------------

        if clean.startswith("-"):

            clean = clean.lstrip(
                "- "
            ).strip()

        if current_detail:

            if current_detail["text"]:

                current_detail["text"] += (
                    " " + clean
                )

            else:

                current_detail["text"] = clean

        elif current_target:

            # For bullet lists such as:
            # - High academic achievers
            # - Students involved in clubs

            current_target[
                "details"
            ].append(
                {
                    "label": "Point",
                    "text": clean
                }
            )

    # --------------------------------------------------------
    # Save final strategy
    # --------------------------------------------------------

    if current_strategy is not None:

        strategies.append(
            current_strategy
        )

    return strategies


# ============================================================
# CREWAI OUTPUT RENDERING
# ============================================================

def render_crewai_strategy(text):

    strategies = parse_crewai_strategy(
        text
    )

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if not strategies:

        st.html(
            f"""
            <div class="strategy-card">

                <div class="strategy-text">
                    {escape(
                        clean_text(text)
                    )}
                </div>

            </div>
            """
        )

        return


    # --------------------------------------------------------
    # RENDER EACH MAJOR STRATEGY
    # --------------------------------------------------------

    for strategy in strategies:

        strategy_html = f"""
        <div class="strategy-card">

            <div class="strategy-number">
                STRATEGY {escape(
                    strategy["number"]
                )}
            </div>

            <div class="strategy-title">
                {escape(
                    strategy["title"]
                )}
            </div>
        """

        # ----------------------------------------------------
        # RENDER TARGETS
        # ----------------------------------------------------

        for target in strategy["targets"]:

            strategy_html += f"""
                <div class="strategy-subtitle">
                    {escape(
                        target["title"]
                    )}
                </div>
            """

            # ------------------------------------------------
            # RENDER DETAILS
            # ------------------------------------------------

            for detail in target["details"]:

                label = detail["label"]
                content = detail["text"]

                # Different styling for actual actions
                if label.lower() == "strategy":

                    strategy_html += f"""
                        <div class="strategy-text">

                            <strong>
                                🎯 Strategy:
                            </strong>

                            {escape(content)}

                        </div>
                    """

                elif label.lower() == "implementation":

                    strategy_html += f"""
                        <div class="strategy-text">

                            <strong>
                                ⚙️ Implementation:
                            </strong>

                            {escape(content)}

                        </div>
                    """

                elif label.lower() == "expected outcome":

                    strategy_html += f"""
                        <div class="strategy-text">

                            <strong>
                                📈 Expected Outcome:
                            </strong>

                            {escape(content)}

                        </div>
                    """

                else:

                    strategy_html += f"""
                        <div class="strategy-text">

                            • {escape(content)}

                        </div>
                    """

        strategy_html += """
        </div>
        """

        st.html(
            strategy_html
        )

# ============================================================
# PDF GENERATION
# ============================================================
def build_pdf(
    department_name,
    student_count,
    avg_gpa,
    avg_attendance,
    avg_engagement,
    gemini_text,
    crewai_text
):

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        leading=25,
        spaceAfter=10
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10,
        textColor=colors.grey,
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        spaceBefore=14,
        spaceAfter=8
    )

    subheading_style = ParagraphStyle(
        "SubHeading",
        parent=styles["Heading3"],
        fontSize=11,
        leading=14,
        spaceBefore=8,
        spaceAfter=5
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=14,
        spaceAfter=6
    )

    story = []

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Student Enrollment & Retention Strategy",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Department Analysis Report — "
            f"{escape(str(department_name))}",
            subtitle_style
        )
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Department Performance Overview",
            heading_style
        )
    )

    metric_data = [
        ["Metric", "Value"],
        ["Students", str(student_count)],
        ["Average GPA", str(avg_gpa)],
        [
            "Average Attendance",
            f"{avg_attendance}%"
        ],
        [
            "Average Engagement",
            f"{avg_engagement}/3"
        ]
    ]

    metric_table = Table(
        metric_data,
        colWidths=[
            3.2 * inch,
            2.5 * inch
        ]
    )

    metric_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#3157c7")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),
            (
                "FONTNAME",
                (0, 1),
                (-1, -1),
                "Helvetica"
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#d9dee8")
            ),
            (
                "BACKGROUND",
                (0, 1),
                (-1, -1),
                colors.whitesmoke
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                8
            )
        ])
    )

    story.append(metric_table)

    # --------------------------------------------------------
    # GEMINI SECTION
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "AI Department Insights",
            heading_style
        )
    )

    sections = split_gemini_sections(
        gemini_text
    )

    for heading in [
        "Key Strengths",
        "Key Risks",
        "Recommended Actions",
        "Overall Assessment"
    ]:

        story.append(
            Paragraph(
                heading,
                subheading_style
            )
        )

        section_text = sections.get(
            heading,
            ""
        )

        items = extract_bullets(
            section_text
        )

        if items:

            for item in items:

                story.append(
                    Paragraph(
                        f"• {escape(item)}",
                        body_style
                    )
                )

        else:

            story.append(
                Paragraph(
                    escape(
                        clean_text(section_text)
                    ),
                    body_style
                )
            )

    # --------------------------------------------------------
    # CREWAI
    # --------------------------------------------------------

           # --------------------------------------------------------
        # CREWAI
        # --------------------------------------------------------
        
        story.append(
            PageBreak()
        )
        
        story.append(
            Paragraph(
                "AI Retention & Enrollment Strategy",
                heading_style
            )
        )
        
        strategies = parse_crewai_strategy(
            crewai_text
        )
        
        if strategies:
        
            for strategy in strategies:
        
                # -----------------------------------------------
                # MAJOR STRATEGY
                # -----------------------------------------------
        
                story.append(
                    Paragraph(
                        f"Strategy {escape(str(strategy['number']))}: "
                        f"{escape(str(strategy['title']))}",
                        subheading_style
                    )
                )
        
                # -----------------------------------------------
                # TARGETS / SUBSECTIONS
                # -----------------------------------------------
        
                for target in strategy.get("targets", []):
        
                    target_title = target.get(
                        "title",
                        "Recommended Approach"
                    )
        
                    story.append(
                        Paragraph(
                            escape(str(target_title)),
                            subheading_style
                        )
                    )
        
                    # -------------------------------------------
                    # DETAILS
                    # -------------------------------------------
        
                    for detail in target.get("details", []):
        
                        label = detail.get(
                            "label",
                            "Point"
                        )
        
                        content = detail.get(
                            "text",
                            ""
                        )
        
                        if not content:
                            continue
        
                        # Strategy / Implementation /
                        # Expected Outcome
                        story.append(
                            Paragraph(
                                f"<b>{escape(str(label))}:</b> "
                                f"{escape(str(content))}",
                                body_style
                            )
                        )
        
        else:
        
            # -----------------------------------------------
            # FALLBACK
            # -----------------------------------------------
        
            story.append(
                Paragraph(
                    escape(
                        clean_text(
                            crewai_text
                        )
                    ),
                    body_style
                )
            )
            document.build(story)
        
            buffer.seek(0)
        
            return buffer.getvalue()


# ============================================================
# MAIN HEADER
# ============================================================

st.html("""
<div class="main-header">

    <div class="main-title">
        🎓 Student Enrollment & Retention Strategy
    </div>

    <div class="main-subtitle">
        Transform department-level student data into actionable
        academic, retention, and enrollment insights using
        Gemini + CrewAI.
    </div>

</div>
""")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🎯 Analysis Center")

    st.write(
        "Upload your department CSV and select a department "
        "to generate an AI-powered performance and retention "
        "strategy."
    )

    st.divider()

    st.markdown("### 📌 Workflow")

    st.markdown("""
    **1.** Upload CSV  
    **2.** Select department  
    **3.** Review performance  
    **4.** Generate AI insights  
    **5.** Review retention strategy  
    **6.** Download PDF report
    """)

    st.divider()

    st.caption(
        "Powered by Gemini + LangChain + CrewAI"
    )


# ============================================================
# FILE UPLOAD
# ============================================================

st.html(
    '<div class="section-heading">'
    '📂 Upload Department Data'
    '</div>'
)

uploaded_file = st.file_uploader(
    "Upload a CSV file containing student department data",
    type=["csv"],
    help=(
        "CSV should contain Department, GPA, "
        "Attendance and Engagement columns."
    )
)


if uploaded_file is None:

    st.info(
        "👋 Upload a CSV file to begin the analysis."
    )

    st.stop()


# ============================================================
# READ CSV
# ============================================================

try:

    df = pd.read_csv(
        uploaded_file
    )

except Exception as e:

    st.error(
        f"❌ Unable to read the CSV file: {e}"
    )

    st.stop()


# ============================================================
# VALIDATE COLUMNS
# ============================================================

required_columns = [
    "Department",
    "GPA",
    "Attendance",
    "Engagement"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    st.error(
        "❌ Missing required columns: "
        + ", ".join(missing_columns)
    )

    st.info(
        "Required columns: Department, GPA, "
        "Attendance, Engagement"
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

st.html(
    '<div class="section-heading">'
    '📋 Dataset Overview'
    '</div>'
)

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

st.html(
    '<div class="section-heading">'
    '🏢 Department Analysis'
    '</div>'
)

department_name = st.selectbox(
    "Select a department",
    sorted(
        df["Department"]
        .dropna()
        .unique()
    ),
    help=(
        "Choose the department you want to analyze."
    )
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze = st.button(
    "🚀 Analyze Department",
    type="primary",
    use_container_width=True
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze:

    # --------------------------------------------------------
    # FILTER DEPARTMENT
    # --------------------------------------------------------

    dept_df = df[
        df["Department"] == department_name
    ]


    # --------------------------------------------------------
    # CALCULATE METRICS
    # --------------------------------------------------------

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

    student_count = len(
        dept_df
    )


    # --------------------------------------------------------
    # DEPARTMENT HEADER
    # --------------------------------------------------------

    st.html(
        f"""
        <div class="department-header">

            <div class="department-badge">
                🏢 {escape(str(department_name))}
            </div>

            <div class="department-title">
                Department Performance Overview
            </div>

            <div class="department-description">
                Analysis based on {student_count}
                student records.
            </div>

        </div>
        """
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metric1, metric2, metric3, metric4 = st.columns(4)


    with metric1:

        st.html(
            f"""
            <div class="metric-card">

                <div class="metric-title">
                    📚 Average GPA
                </div>

                <div class="metric-value">
                    {avg_gpa}
                </div>

            </div>
            """
        )


    with metric2:

        st.html(
            f"""
            <div class="metric-card">

                <div class="metric-title">
                    📅 Attendance
                </div>

                <div class="metric-value">
                    {avg_attendance}%
                </div>

            </div>
            """
        )


    with metric3:

        st.html(
            f"""
            <div class="metric-card">

                <div class="metric-title">
                    📈 Engagement
                </div>

                <div class="metric-value">
                    {avg_engagement}/3
                </div>

            </div>
            """
        )


    with metric4:

        st.html(
            f"""
            <div class="metric-card">

                <div class="metric-title">
                    👥 Students
                </div>

                <div class="metric-value">
                    {student_count}
                </div>

            </div>
            """
        )


    # ========================================================
    # GEMINI ANALYSIS
    # ========================================================

    st.html(
        '<div class="section-heading">'
        '🧠 AI Department Insights'
        '</div>'
    )

    st.html(
        """
        <div class="section-description">
            Gemini analyzes academic performance, attendance,
            engagement and potential retention risks.
        </div>
        """
    )


    try:

        with st.spinner(
            "🧠 Gemini is analyzing department performance..."
        ):

            dept_result = dept_chain.invoke(
                {
                    "Department": department_name,
                    "Avg_GPA": avg_gpa,
                    "Avg_Attendance": avg_attendance,
                    "Avg_Engagement": avg_engagement
                }
            )


            # ------------------------------------------------
            # HANDLE DIFFERENT LANGCHAIN RESPONSE FORMATS
            # ------------------------------------------------

            if isinstance(
                dept_result,
                dict
            ):

                dept_summary = dept_result.get(
                    "text",
                    ""
                )

            else:

                dept_summary = str(
                    dept_result
                )


            # Handle AIMessage-like objects
            if hasattr(
                dept_summary,
                "content"
            ):

                dept_summary = (
                    dept_summary.content
                )


            dept_summary = str(
                dept_summary
            ).strip()


        render_gemini_output(
            dept_summary
        )


    except Exception as e:

        st.error(
            f"❌ Gemini analysis failed: {e}"
        )

        st.stop()


    # ========================================================
    # CREWAI STRATEGY
    # ========================================================

    st.html(
        '<div class="section-heading">'
        '🤖 AI Retention & Enrollment Strategy'
        '</div>'
    )

    st.html(
        """
        <div class="section-description">
            CrewAI generates targeted strategies for improving
            student retention and strengthening enrollment.
        </div>
        """
    )


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


        render_crewai_strategy(
            analysis
        )


    except Exception as e:

        st.error(
            f"❌ CrewAI request failed: {e}"
        )

        analysis = (
            "CrewAI strategy could not be generated."
        )


    # ========================================================
    # PDF DOWNLOAD
    # ========================================================

    st.html(
        '<div class="section-heading">'
        '📄 Download Report'
        '</div>'
    )

    st.html(
        """
        <div class="download-card">

            <div style="font-size:1.2rem;font-weight:700;">
                Complete Department Report
            </div>

            <div style="color:#667085;margin-top:0.4rem;">
                Download the department metrics, Gemini
                insights and CrewAI retention strategy as a PDF.
            </div>

        </div>
        """
    )


    try:

        pdf_data = build_pdf(
            department_name=department_name,
            student_count=student_count,
            avg_gpa=avg_gpa,
            avg_attendance=avg_attendance,
            avg_engagement=avg_engagement,
            gemini_text=dept_summary,
            crewai_text=analysis
        )


        st.download_button(
            label="⬇️ Download Complete Report (PDF)",
            data=pdf_data,
            file_name=(
                f"{department_name}_"
                "Retention_Strategy_Report.pdf"
            ),
            mime="application/pdf",
            use_container_width=True
        )


    except Exception as e:

        st.error(
            f"❌ Could not generate PDF: {e}"
        )


    # ========================================================
    # SUCCESS
    # ========================================================

    st.success(
        f"✅ Analysis completed successfully "
        f"for {department_name}."
    )


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div class="footer">

        Student Enrollment & Retention Strategy Dashboard

        <br><br>

        Gemini • LangChain • CrewAI • Streamlit

    </div>
    """
)
