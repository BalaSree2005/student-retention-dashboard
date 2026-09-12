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
# CUSTOM CSS — MODERN AI DASHBOARD UI
# ============================================================

st.html("""
<style>

/* ============================================================
   GLOBAL APP BACKGROUND
   ============================================================ */

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(99, 102, 241, 0.12),
            transparent 28%
        ),
        radial-gradient(
            circle at 90% 15%,
            rgba(14, 165, 233, 0.10),
            transparent 25%
        ),
        radial-gradient(
            circle at 50% 90%,
            rgba(168, 85, 247, 0.08),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #f8fafc 0%,
            #eef2ff 45%,
            #f8fafc 100%
        );

    background-attachment: fixed;
}


/* ============================================================
   ANIMATED BACKGROUND GLOW
   ============================================================ */

.stApp::before {
    content: "";
    position: fixed;

    width: 420px;
    height: 420px;

    top: -160px;
    right: -120px;

    background:
        radial-gradient(
            circle,
            rgba(99, 102, 241, 0.20),
            transparent 68%
        );

    border-radius: 50%;

    filter: blur(20px);

    animation: floatingGlow 8s ease-in-out infinite;

    pointer-events: none;

    z-index: 0;
}

.stApp::after {
    content: "";

    position: fixed;

    width: 350px;
    height: 350px;

    bottom: -140px;
    left: -100px;

    background:
        radial-gradient(
            circle,
            rgba(14, 165, 233, 0.15),
            transparent 70%
        );

    border-radius: 50%;

    filter: blur(25px);

    animation: floatingGlow2 10s ease-in-out infinite;

    pointer-events: none;

    z-index: 0;
}


@keyframes floatingGlow {

    0%, 100% {
        transform: translate(0, 0) scale(1);
    }

    50% {
        transform: translate(-30px, 35px) scale(1.12);
    }
}


@keyframes floatingGlow2 {

    0%, 100% {
        transform: translate(0, 0) scale(1);
    }

    50% {
        transform: translate(35px, -25px) scale(1.08);
    }
}


/* ============================================================
   MAIN CONTAINER
   ============================================================ */

.block-container {

    padding-top: 2.5rem;
    padding-bottom: 4rem;

    max-width: 1450px;

    position: relative;
    z-index: 1;
}


/* ============================================================
   HEADER
   ============================================================ */

.main-header {

    position: relative;

    padding: 2.5rem 2.8rem;

    margin-bottom: 2rem;

    border-radius: 28px;

    overflow: hidden;

    background:
        linear-gradient(
            135deg,
            rgba(30, 41, 59, 0.98),
            rgba(49, 46, 129, 0.96),
            rgba(79, 70, 229, 0.94)
        );

    box-shadow:
        0 25px 60px rgba(49, 46, 129, 0.20);

    animation: headerEntrance 0.8s ease-out;
}


.main-header::before {

    content: "";

    position: absolute;

    width: 320px;
    height: 320px;

    right: -80px;
    top: -140px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(255,255,255,0.20),
            transparent 65%
        );

    animation: headerOrb 7s ease-in-out infinite;
}


.main-header::after {

    content: "";

    position: absolute;

    width: 200px;
    height: 200px;

    left: 40%;

    bottom: -130px;

    border-radius: 50%;

    border: 1px solid rgba(255,255,255,0.15);

    box-shadow:
        0 0 60px rgba(255,255,255,0.08);
}


@keyframes headerEntrance {

    from {
        opacity: 0;
        transform: translateY(-20px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}


@keyframes headerOrb {

    0%, 100% {
        transform: translate(0,0);
    }

    50% {
        transform: translate(-35px,35px);
    }
}


.main-title {

    position: relative;

    z-index: 2;

    font-size: 2.55rem;

    font-weight: 800;

    letter-spacing: -0.8px;

    color: #ffffff;

    margin-bottom: 0.65rem;

    text-shadow:
        0 4px 20px rgba(0,0,0,0.18);
}


.main-subtitle {

    position: relative;

    z-index: 2;

    font-size: 1.02rem;

    line-height: 1.7;

    color: rgba(255,255,255,0.82);

    max-width: 900px;
}


/* ============================================================
   SECTION HEADINGS
   ============================================================ */

.section-heading {

    position: relative;

    font-size: 1.45rem;

    font-weight: 800;

    color: #172033;

    margin-top: 2rem;

    margin-bottom: 0.8rem;

    padding-left: 15px;
}


.section-heading::before {

    content: "";

    position: absolute;

    left: 0;
    top: 4px;

    width: 5px;
    height: 24px;

    border-radius: 10px;

    background:
        linear-gradient(
            180deg,
            #6366f1,
            #06b6d4
        );

    box-shadow:
        0 0 12px rgba(99,102,241,0.35);
}


.section-description {

    color: #64748b;

    font-size: 0.93rem;

    line-height: 1.6;

    margin-bottom: 1.1rem;
}


/* ============================================================
   METRIC CARDS
   ============================================================ */

.metric-card {

    position: relative;

    min-height: 135px;

    padding: 1.35rem 1.4rem;

    border-radius: 20px;

    background:
        rgba(255,255,255,0.88);

    backdrop-filter: blur(14px);

    border: 1px solid rgba(255,255,255,0.85);

    box-shadow:
        0 10px 30px rgba(15,23,42,0.07);

    overflow: hidden;

    transition:
        transform 0.35s ease,
        box-shadow 0.35s ease,
        border-color 0.35s ease;
}


.metric-card::before {

    content: "";

    position: absolute;

    width: 90px;
    height: 90px;

    right: -35px;
    top: -35px;

    border-radius: 50%;

    background:
        linear-gradient(
            135deg,
            rgba(99,102,241,0.16),
            rgba(6,182,212,0.08)
        );
}


.metric-card:hover {

    transform:
        translateY(-8px)
        scale(1.015);

    box-shadow:
        0 20px 45px rgba(15,23,42,0.13);

    border-color:
        rgba(99,102,241,0.35);
}


.metric-title {

    color: #64748b;

    font-size: 0.82rem;

    font-weight: 700;

    text-transform: uppercase;

    letter-spacing: 0.7px;

    margin-bottom: 0.65rem;
}


.metric-value {

    color: #111827;

    font-size: 2rem;

    font-weight: 850;

    letter-spacing: -1px;
}


/* ============================================================
   DEPARTMENT HEADER
   ============================================================ */

.department-header {

    position: relative;

    padding: 1.5rem 1.7rem;

    margin-top: 1.4rem;

    margin-bottom: 1.2rem;

    border-radius: 22px;

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.95),
            rgba(238,242,255,0.9)
        );

    border: 1px solid #dbe4ff;

    box-shadow:
        0 12px 35px rgba(49,46,129,0.08);

    overflow: hidden;
}


.department-header::after {

    content: "";

    position: absolute;

    width: 180px;
    height: 180px;

    right: -80px;
    top: -90px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(99,102,241,0.15),
            transparent 68%
        );
}


.department-badge {

    display: inline-flex;

    align-items: center;

    background:
        linear-gradient(
            135deg,
            #eef2ff,
            #e0f2fe
        );

    color: #4338ca;

    padding: 0.42rem 0.9rem;

    border-radius: 999px;

    font-size: 0.78rem;

    font-weight: 800;

    border: 1px solid #c7d2fe;

    margin-bottom: 0.65rem;
}


.department-title {

    font-size: 1.35rem;

    font-weight: 800;

    color: #172033;
}


.department-description {

    color: #64748b;

    font-size: 0.9rem;

    margin-top: 0.25rem;
}


/* ============================================================
   STRENGTH CARDS — GREEN / SUCCESS
   ============================================================ */

.strength-card {

    position: relative;

    padding: 1.25rem 1.35rem 1.25rem 4.2rem;

    margin-bottom: 1rem;

    border-radius: 18px;

    background:
        linear-gradient(
            135deg,
            #f0fdf4,
            #ecfdf5
        );

    border: 1px solid #bbf7d0;

    box-shadow:
        0 8px 25px rgba(34,197,94,0.07);

    transition:
        transform 0.3s ease,
        box-shadow 0.3s ease;
}


.strength-card::before {

    content: "✓";

    position: absolute;

    left: 17px;
    top: 50%;

    transform: translateY(-50%);

    width: 42px;
    height: 42px;

    display: flex;

    align-items: center;
    justify-content: center;

    border-radius: 14px;

    background:
        linear-gradient(
            135deg,
            #22c55e,
            #16a34a
        );

    color: white;

    font-size: 1.2rem;

    font-weight: 900;

    box-shadow:
        0 8px 18px rgba(34,197,94,0.25);
}


.strength-card:hover {

    transform: translateX(7px);

    box-shadow:
        0 14px 30px rgba(34,197,94,0.13);
}


.strength-title {

    color: #166534;

    font-weight: 800;

    font-size: 1rem;

    margin-bottom: 0.4rem;
}


/* ============================================================
   RISK CARDS — ORANGE / WARNING
   ============================================================ */

.risk-card {

    position: relative;

    padding: 1.2rem 1.3rem 1.2rem 4.3rem;

    margin-bottom: 0.9rem;

    border-radius: 16px;

    background:
        linear-gradient(
            135deg,
            #fff7ed,
            #fffbeb
        );

    border: 1px solid #fed7aa;

    transition:
        transform 0.3s ease,
        box-shadow 0.3s ease;
}


.risk-card::before {

    content: "⚠";

    position: absolute;

    left: 16px;
    top: 50%;

    transform: translateY(-50%);

    width: 43px;
    height: 43px;

    display: flex;

    align-items: center;
    justify-content: center;

    border-radius: 50%;

    background:
        linear-gradient(
            135deg,
            #f97316,
            #ea580c
        );

    color: white;

    font-size: 1.05rem;

    box-shadow:
        0 7px 18px rgba(249,115,22,0.22);
}


.risk-card:hover {

    transform: translateX(7px);

    box-shadow:
        0 14px 30px rgba(249,115,22,0.13);
}


.risk-title {

    color: #9a3412;

    font-weight: 800;

    font-size: 0.98rem;

    margin-bottom: 0.35rem;
}


/* ============================================================
   ACTION CARDS — BLUE
   ============================================================ */

.action-card {

    position: relative;

    padding: 1.25rem 1.35rem 1.25rem 4.8rem;

    margin-bottom: 0.9rem;

    border-radius: 18px;

    background:
        #ffffff;

    border: 1px solid #dbeafe;

    box-shadow:
        0 7px 24px rgba(37,99,235,0.06);

    transition:
        transform 0.3s ease,
        box-shadow 0.3s ease;
}


.action-card::before {

    content: attr(data-number);

    position: absolute;

    left: 17px;
    top: 20px;

    width: 45px;
    height: 45px;

    border-radius: 14px;

    display: flex;

    align-items: center;
    justify-content: center;

    background:
        linear-gradient(
            135deg,
            #3b82f6,
            #4f46e5
        );

    color: white;

    font-size: 0.85rem;

    font-weight: 900;

    box-shadow:
        0 8px 20px rgba(59,130,246,0.23);
}


.action-card:hover {

    transform:
        translateY(-5px);

    box-shadow:
        0 16px 35px rgba(37,99,235,0.12);

    border-color:
        #93c5fd;
}


.action-number {

    color: #2563eb;

    font-size: 0.72rem;

    font-weight: 850;

    letter-spacing: 1px;

    margin-bottom: 0.2rem;
}


.action-title {

    color: #172033;

    font-size: 1.03rem;

    font-weight: 800;

    margin-bottom: 0.35rem;
}


.action-description {

    color: #475569;

    font-size: 0.9rem;

    line-height: 1.6;
}


/* ============================================================
   OVERALL ASSESSMENT
   ============================================================ */

.assessment-card {

    position: relative;

    padding: 1.6rem 1.7rem;

    border-radius: 22px;

    background:
        linear-gradient(
            135deg,
            #eef2ff,
            #f0f9ff
        );

    border: 1px solid #c7d2fe;

    box-shadow:
        0 12px 35px rgba(79,70,229,0.09);

    overflow: hidden;
}


.assessment-card::after {

    content: "";

    position: absolute;

    right: -50px;
    bottom: -80px;

    width: 190px;
    height: 190px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(99,102,241,0.14),
            transparent 68%
        );
}


.assessment-title {

    color: #3730a3;

    font-size: 1.08rem;

    font-weight: 850;

    margin-bottom: 0.45rem;
}


.assessment-text {

    color: #334155;

    font-size: 0.93rem;

    line-height: 1.7;
}


/* ============================================================
   CREWAI STRATEGY CARDS
   ============================================================ */

.strategy-card {

    position: relative;

    padding: 1.7rem 1.7rem 1.6rem 1.8rem;

    margin-bottom: 1.25rem;

    border-radius: 24px;

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.96),
            rgba(248,250,252,0.92)
        );

    border: 1px solid #e2e8f0;

    box-shadow:
        0 12px 35px rgba(15,23,42,0.07);

    overflow: hidden;

    transition:
        transform 0.35s ease,
        box-shadow 0.35s ease,
        border-color 0.35s ease;
}


.strategy-card::before {

    content: "";

    position: absolute;

    left: 0;
    top: 0;

    width: 6px;
    height: 100%;

    background:
        linear-gradient(
            180deg,
            #6366f1,
            #8b5cf6,
            #06b6d4
        );
}


.strategy-card::after {

    content: "";

    position: absolute;

    width: 170px;
    height: 170px;

    right: -70px;
    top: -70px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(99,102,241,0.10),
            transparent 70%
        );

    transition:
        transform 0.5s ease;
}


.strategy-card:hover {

    transform:
        translateY(-7px);

    box-shadow:
        0 20px 45px rgba(15,23,42,0.12);

    border-color:
        #c7d2fe;
}


.strategy-card:hover::after {

    transform:
        scale(1.4);
}


.strategy-number {

    display: inline-flex;

    padding: 0.35rem 0.75rem;

    border-radius: 999px;

    background:
        #eef2ff;

    color: #4f46e5;

    font-size: 0.72rem;

    font-weight: 850;

    letter-spacing: 0.8px;

    margin-bottom: 0.55rem;
}


.strategy-title {

    color: #111827;

    font-size: 1.15rem;

    font-weight: 850;

    margin-bottom: 1rem;
}


.strategy-subtitle {

    position: relative;

    color: #4338ca;

    font-size: 0.96rem;

    font-weight: 800;

    margin-top: 1rem;

    margin-bottom: 0.4rem;

    padding-left: 14px;
}


.strategy-subtitle::before {

    content: "";

    position: absolute;

    left: 0;
    top: 5px;

    width: 6px;
    height: 6px;

    border-radius: 50%;

    background: #6366f1;

    box-shadow:
        0 0 8px rgba(99,102,241,0.5);
}


.strategy-text {

    color: #475569;

    font-size: 0.91rem;

    line-height: 1.65;

    padding: 0.45rem 0;

    margin-left: 14px;

    border-bottom: 1px dashed #e2e8f0;
}


.strategy-text:last-child {

    border-bottom: none;
}


/* ============================================================
   UPLOADER
   ============================================================ */

[data-testid="stFileUploader"] {

    padding: 1.5rem;

    border-radius: 22px;

    background:
        rgba(255,255,255,0.80);

    backdrop-filter: blur(12px);

    border: 2px dashed #c7d2fe;

    box-shadow:
        0 10px 30px rgba(79,70,229,0.06);

    transition:
        border-color 0.3s ease,
        background 0.3s ease,
        transform 0.3s ease;
}


[data-testid="stFileUploader"]:hover {

    border-color: #6366f1;

    background:
        rgba(238,242,255,0.85);

    transform:
        translateY(-3px);
}


/* ============================================================
   SELECTBOX
   ============================================================ */

div[data-baseweb="select"] > div {

    border-radius: 14px !important;

    border: 1px solid #cbd5e1 !important;

    background: rgba(255,255,255,0.9) !important;

    transition:
        border-color 0.25s ease,
        box-shadow 0.25s ease;
}


div[data-baseweb="select"] > div:hover {

    border-color: #6366f1 !important;

    box-shadow:
        0 0 0 3px rgba(99,102,241,0.10) !important;
}


/* ============================================================
   PRIMARY BUTTON
   ============================================================ */

.stButton > button {

    min-height: 50px;

    border: none;

    border-radius: 15px;

    font-size: 0.95rem;

    font-weight: 800;

    letter-spacing: 0.2px;

    background:
        linear-gradient(
            135deg,
            #4f46e5,
            #6366f1,
            #8b5cf6
        );

    color: white;

    box-shadow:
        0 10px 25px rgba(79,70,229,0.25);

    transition:
        transform 0.25s ease,
        box-shadow 0.25s ease,
        filter 0.25s ease;
}


.stButton > button:hover {

    transform:
        translateY(-3px);

    filter:
        brightness(1.05);

    box-shadow:
        0 16px 35px rgba(79,70,229,0.35);
}


.stButton > button:active {

    transform:
        translateY(0)
        scale(0.98);
}


/* ============================================================
   DOWNLOAD BUTTON
   ============================================================ */

.stDownloadButton > button {

    min-height: 50px;

    border-radius: 14px;

    border: 1px solid #c7d2fe;

    background:
        linear-gradient(
            135deg,
            #eef2ff,
            #e0f2fe
        );

    color: #3730a3;

    font-weight: 800;

    transition:
        transform 0.25s ease,
        box-shadow 0.25s ease;
}


.stDownloadButton > button:hover {

    transform:
        translateY(-3px);

    box-shadow:
        0 12px 28px rgba(79,70,229,0.15);
}


/* ============================================================
   DOWNLOAD SECTION
   ============================================================ */

.download-card {

    position: relative;

    padding: 1.8rem;

    margin-top: 1.2rem;

    text-align: center;

    border-radius: 24px;

    background:
        linear-gradient(
            135deg,
            #ffffff,
            #f5f3ff
        );

    border: 1px solid #ddd6fe;

    box-shadow:
        0 12px 35px rgba(124,58,237,0.08);

    overflow: hidden;
}


.download-card::before {

    content: "📄";

    display: block;

    font-size: 2.2rem;

    margin-bottom: 0.5rem;

    animation:
        floatingIcon 3s ease-in-out infinite;
}


@keyframes floatingIcon {

    0%, 100% {
        transform: translateY(0);
    }

    50% {
        transform: translateY(-6px);
    }
}


/* ============================================================
   EXPANDER
   ============================================================ */

[data-testid="stExpander"] {

    border-radius: 18px !important;

    border: 1px solid #e2e8f0 !important;

    background:
        rgba(255,255,255,0.75) !important;

    box-shadow:
        0 7px 25px rgba(15,23,42,0.05);

    transition:
        box-shadow 0.3s ease;
}


[data-testid="stExpander"]:hover {

    box-shadow:
        0 12px 30px rgba(15,23,42,0.08);
}


/* ============================================================
   DATAFRAME
   ============================================================ */

[data-testid="stDataFrame"] {

    border-radius: 16px;

    overflow: hidden;

    box-shadow:
        0 8px 25px rgba(15,23,42,0.06);
}


/* ============================================================
   SUCCESS MESSAGE
   ============================================================ */

[data-testid="stAlert"] {

    border-radius: 16px !important;

    border: 1px solid rgba(99,102,241,0.15);

    box-shadow:
        0 7px 20px rgba(15,23,42,0.05);
}


/* ============================================================
   SIDEBAR
   ============================================================ */

section[data-testid="stSidebar"] {

    background:
        linear-gradient(
            180deg,
            #111827 0%,
            #1e1b4b 55%,
            #312e81 100%
        );

    border-right: 1px solid rgba(255,255,255,0.08);
}


section[data-testid="stSidebar"] * {

    color: rgba(255,255,255,0.90);
}


section[data-testid="stSidebar"] hr {

    border-color:
        rgba(255,255,255,0.12);
}


section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {

    color: rgba(255,255,255,0.82);
}


/* ============================================================
   SIDEBAR HEADINGS
   ============================================================ */

section[data-testid="stSidebar"] h2 {

    color: white;

    font-weight: 850;
}


section[data-testid="stSidebar"] h3 {

    color: #c7d2fe;

    font-weight: 800;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {

    position: relative;

    text-align: center;

    color: #94a3b8;

    font-size: 0.78rem;

    line-height: 1.7;

    margin-top: 4rem;

    padding: 1.5rem 0;

    border-top: 1px solid rgba(148,163,184,0.25);
}


/* ============================================================
   SCROLLBAR
   ============================================================ */

::-webkit-scrollbar {

    width: 9px;
}


::-webkit-scrollbar-track {

    background: #f1f5f9;
}


::-webkit-scrollbar-thumb {

    background:
        linear-gradient(
            180deg,
            #6366f1,
            #8b5cf6
        );

    border-radius: 10px;
}


::-webkit-scrollbar-thumb:hover {

    background:
        linear-gradient(
            180deg,
            #4f46e5,
            #7c3aed
        );
}


/* ============================================================
   RESPONSIVE DESIGN
   ============================================================ */

@media (max-width: 900px) {

    .main-title {

        font-size: 2rem;
    }

    .main-header {

        padding: 2rem 1.6rem;

        border-radius: 22px;
    }

    .metric-card {

        margin-bottom: 1rem;
    }

}


@media (max-width: 600px) {

    .block-container {

        padding-left: 1rem;

        padding-right: 1rem;
    }

    .main-title {

        font-size: 1.65rem;
    }

    .main-subtitle {

        font-size: 0.9rem;
    }

    .section-heading {

        font-size: 1.2rem;
    }

    .strategy-card {

        padding: 1.3rem;
    }

}


/* ============================================================
   REDUCE MOTION FOR ACCESSIBILITY
   ============================================================ */

@media (prefers-reduced-motion: reduce) {

    *,
    *::before,
    *::after {

        animation-duration: 0.01ms !important;

        animation-iteration-count: 1 !important;

        transition-duration: 0.01ms !important;

        scroll-behavior: auto !important;
    }
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
    Parse CrewAI output into:

    Strategy
        -> Target / Subsection
            -> Detail / Point
    """

    strategies = []

    current_strategy = None
    current_target = None
    current_detail = None

    for raw_line in str(text).splitlines():

        line = raw_line.strip()

        if not line:
            continue

        # ----------------------------------------------------
        # REMOVE MARKDOWN FORMATTING
        # ----------------------------------------------------

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

        line = re.sub(
            r"`(.*?)`",
            r"\1",
            line
        )

        line = line.strip()

        # ----------------------------------------------------
        # IGNORE SEPARATORS
        # ----------------------------------------------------

        if line in ["---", "***", "___"]:
            continue

        # ----------------------------------------------------
        # REMOVE HEADING SYMBOLS
        # ----------------------------------------------------

        clean = re.sub(
            r"^#+\s*",
            "",
            line
        ).strip()

        # ----------------------------------------------------
        # IGNORE REPORT TITLE / OBJECTIVE
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
        #
        # Example:
        # 1. Recruitment Strategies Segmented by GPA Levels
        # 2. Targeting High-Engagement Student Populations
        # ----------------------------------------------------

        major_match = re.match(
            r"^(\d+)\.\s+(.+)$",
            clean
        )

        if major_match:

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
        # REMOVE BULLET SYMBOL FOR DETECTION
        # ----------------------------------------------------

        is_bullet = bool(
            re.match(
                r"^[-•*]\s+",
                clean
            )
        )

        bullet_text = re.sub(
            r"^[-•*]\s+",
            "",
            clean
        ).strip()

        # ----------------------------------------------------
        # LETTERED SUBSECTIONS
        #
        # Example:
        # a. High GPA (3.5 - 4.0)
        # b. Mid GPA (3.0 - 3.49)
        # c. Low GPA (Below 3.0)
        # ----------------------------------------------------

        letter_target = re.match(
            r"^[a-zA-Z]\.\s+(.+)$",
            clean
        )

        if letter_target and current_strategy:

            current_target = {
                "title": letter_target.group(1).strip(),
                "details": []
            }

            current_strategy["targets"].append(
                current_target
            )

            current_detail = None

            continue

        # ----------------------------------------------------
        # KNOWN TARGET / SUBSECTION HEADINGS
        # ----------------------------------------------------

        target_patterns = [
            r"^High GPA.*",
            r"^Mid GPA.*",
            r"^Medium GPA.*",
            r"^Moderate GPA.*",
            r"^Low GPA.*",

            r"^High[- ]Engagement.*",
            r"^Medium[- ]Engagement.*",
            r"^Moderate[- ]Engagement.*",
            r"^Low[- ]Engagement.*",

            r"^Engagement Strategy.*",
            r"^Attendance Initiative.*",
            r"^Attendance Strategy.*",

            r"^Partnership Strategy.*",
            r"^Partnership Initiative.*",

            r"^Successful Segments.*",
            r"^At-Risk Segments.*",
            r"^Demographic Profiles.*",

            r"^Conclusion:?$"
        ]

        is_target = any(
            re.match(
                pattern,
                clean,
                re.IGNORECASE
            )
            for pattern in target_patterns
        )

        if (
            is_target
            and current_strategy
            and not is_bullet
        ):

            current_target = {
                "title": clean.rstrip(":"),
                "details": []
            }

            current_strategy["targets"].append(
                current_target
            )

            current_detail = None

            continue

        # ----------------------------------------------------
        # DETAIL WITH LABEL
        #
        # Example:
        # Strategy: ...
        # Implementation: ...
        # Expected Outcome: ...
        #
        # Also supports:
        # Targeted Outreach Campaign: ...
        # Incentives: ...
        # Expected Outcomes: ...
        # ----------------------------------------------------

        detail_match = re.match(
            r"^([^:]{2,80}):\s*(.*)$",
            bullet_text
        )

        if (
            detail_match
            and current_strategy
        ):

            label = detail_match.group(1).strip()
            content = detail_match.group(2).strip()

            # Create target automatically if needed
            if current_target is None:

                current_target = {
                    "title": "Recommended Approach",
                    "details": []
                }

                current_strategy["targets"].append(
                    current_target
                )

            detail = {
                "label": label,
                "text": content
            }

            current_target["details"].append(
                detail
            )

            current_detail = detail

            continue

        # ----------------------------------------------------
        # NORMAL BULLET / POINT
        #
        # This is important for Strategies 2, 3 and 4.
        # If they don't have a subsection heading,
        # automatically create "Recommended Approach".
        # ----------------------------------------------------

        if is_bullet:

            if current_strategy is None:
                continue

            if current_target is None:

                current_target = {
                    "title": "Recommended Approach",
                    "details": []
                }

                current_strategy["targets"].append(
                    current_target
                )

            current_target["details"].append(
                {
                    "label": "Point",
                    "text": bullet_text
                }
            )

            current_detail = None

            continue

        # ----------------------------------------------------
        # CONTINUATION OF PREVIOUS DETAIL
        # ----------------------------------------------------

        if current_detail:

            if current_detail["text"]:

                current_detail["text"] += (
                    " " + clean
                )

            else:

                current_detail["text"] = clean

            continue

        # ----------------------------------------------------
        # STANDALONE TEXT
        #
        # If a strategy has text but no target,
        # create a default target.
        # ----------------------------------------------------

        if current_strategy:

            if current_target is None:

                current_target = {
                    "title": "Recommended Approach",
                    "details": []
                }

                current_strategy["targets"].append(
                    current_target
                )

            current_target["details"].append(
                {
                    "label": "Point",
                    "text": clean
                }
            )

    # --------------------------------------------------------
    # SAVE FINAL STRATEGY
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

    strategies = parse_crewai_strategy(text)

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if not strategies:

        st.html(
            f"""
            <div class="strategy-card">

                <div class="strategy-text">
                    {escape(clean_text(text))}
                </div>

            </div>
            """
        )

        return

    # --------------------------------------------------------
    # RENDER EACH STRATEGY
    # --------------------------------------------------------

    for strategy in strategies:

        strategy_number = strategy.get(
            "number",
            ""
        )

        strategy_title = strategy.get(
            "title",
            "Retention Strategy"
        )

        strategy_html = f"""
        <div class="strategy-card">

            <div class="strategy-number">
                STRATEGY {escape(str(strategy_number))}
            </div>

            <div class="strategy-title">
                {escape(str(strategy_title))}
            </div>
        """

        # ----------------------------------------------------
        # TARGETS
        # ----------------------------------------------------

        for target in strategy.get(
            "targets",
            []
        ):

            target_title = target.get(
                "title",
                "Recommended Approach"
            )

            strategy_html += f"""
                <div class="strategy-subtitle">
                    {escape(str(target_title))}
                </div>
            """

            # ------------------------------------------------
            # DETAILS
            # ------------------------------------------------

            for detail in target.get(
                "details",
                []
            ):

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

                # --------------------------------------------
                # POINT WITHOUT LABEL
                # --------------------------------------------

                if label.lower() == "point":

                    strategy_html += f"""
                        <div class="strategy-text">
                            • {escape(str(content))}
                        </div>
                    """

                # --------------------------------------------
                # LABELED DETAIL
                # --------------------------------------------

                else:

                    strategy_html += f"""
                        <div class="strategy-text">

                            <strong>
                                {escape(str(label))}:
                            </strong>

                            {escape(str(content))}

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

    # --------------------------------------------------------
    # PDF STYLES
    # --------------------------------------------------------

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

    # ========================================================
    # TITLE
    # ========================================================

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

    # ========================================================
    # DEPARTMENT METRICS
    # ========================================================

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

    # ========================================================
    # GEMINI SECTION
    # ========================================================

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
                        f"• {escape(str(item))}",
                        body_style
                    )
                )

        else:

            cleaned_section = clean_text(
                section_text
            )

            if cleaned_section:

                story.append(
                    Paragraph(
                        escape(
                            str(cleaned_section)
                        ),
                        body_style
                    )
                )

            else:

                story.append(
                    Paragraph(
                        "No information available.",
                        body_style
                    )
                )

    # ========================================================
    # CREWAI STRATEGY
    # ========================================================

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

    # --------------------------------------------------------
    # PARSED CREWAI STRATEGY
    # --------------------------------------------------------

    if strategies:

        for strategy in strategies:

            # -----------------------------------------------
            # MAJOR STRATEGY
            # -----------------------------------------------

            strategy_number = strategy.get(
                "number",
                ""
            )

            strategy_title = strategy.get(
                "title",
                "Retention Strategy"
            )

            story.append(
                Paragraph(
                    f"Strategy "
                    f"{escape(str(strategy_number))}: "
                    f"{escape(str(strategy_title))}",
                    subheading_style
                )
            )

            # -----------------------------------------------
            # TARGETS / SUBSECTIONS
            # -----------------------------------------------

            for target in strategy.get(
                "targets",
                []
            ):

                target_title = target.get(
                    "title",
                    "Recommended Approach"
                )

                story.append(
                    Paragraph(
                        escape(
                            str(target_title)
                        ),
                        subheading_style
                    )
                )

                # -------------------------------------------
                # DETAILS
                # -------------------------------------------

                for detail in target.get(
                    "details",
                    []
                ):

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

                    story.append(
                        Paragraph(
                            f"<b>{escape(str(label))}:</b> "
                            f"{escape(str(content))}",
                            body_style
                        )
                    )

    # --------------------------------------------------------
    # FALLBACK IF CREWAI PARSING FAILS
    # --------------------------------------------------------

    else:

        cleaned_crewai = clean_text(
            crewai_text
        )

        if cleaned_crewai:

            # Split into lines so the PDF is easier to read
            for line in str(
                cleaned_crewai
            ).splitlines():

                line = line.strip()

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

                line = re.sub(
                    r"`(.*?)`",
                    r"\1",
                    line
                )

                # Remove markdown heading symbols
                line = re.sub(
                    r"^#+\s*",
                    "",
                    line
                )

                # Handle bullet points
                if line.startswith("-"):

                    line = line[1:].strip()

                    story.append(
                        Paragraph(
                            f"• {escape(str(line))}",
                            body_style
                        )
                    )

                elif line.startswith("•"):

                    story.append(
                        Paragraph(
                            escape(str(line)),
                            body_style
                        )
                    )

                else:

                    story.append(
                        Paragraph(
                            escape(str(line)),
                            body_style
                        )
                    )

        else:

            story.append(
                Paragraph(
                    "No CrewAI retention strategy "
                    "was available.",
                    body_style
                )
            )

    # ========================================================
    # BUILD PDF
    # ========================================================

    # IMPORTANT:
    # These lines MUST be outside the if/else above.
    # Otherwise the function can return None.

    document.build(
        story
    )

    buffer.seek(0)

    pdf_data = buffer.getvalue()

    # Safety check
    if not pdf_data:

        raise ValueError(
            "PDF was generated but contains no data."
        )

    return pdf_data


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
