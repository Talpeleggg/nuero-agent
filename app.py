import streamlit as st
import pandas as pd
import os
import glob

# Import our modular agent factory!
from agent import get_neural_agent

# 1. High-End Page Configuration
st.set_page_config(
    page_title="NeuroData Pipeline",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CUSTOM STYLING
# ---------------------------------------------------------
st.markdown("""
    <style>
    .hero-banner {
        background: linear-gradient(135deg, #101015 0%, #202025 100%);
        padding: 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid #303035;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::after {
        content: '';
        position: absolute;
        top: 0; right: 0; width: 300px; height: 100%;
        background-image: url('https://img.icons8.com/plasticine/200/brain.png');
        background-repeat: no-repeat;
        background-position: center;
        opacity: 0.15;
    }
    .hero-title {
        color: #FFFFFF;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
    }
    .hero-subtitle {
        color: #A0A0A5;
        font-size: 1.1rem;
        margin-bottom: 0px !important;
    }
    .stChatMessage { border-radius: 10px; margin-bottom: 10px; }
    .stChatMessage.user { background-color: #262730; }
    .stChatMessage.assistant { background-color: #101015; border: 1px solid #202025; }
    div.stMetric { background-color: #1A1A20; border-radius: 10px; padding: 15px; border: 1px solid #202025; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# THE ARTISTIC BRAIN BAR (Header)
# ---------------------------------------------------------
st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">🧠 NeuroData Pipeline</h1>
        <p class="hero-subtitle">High-Performance Automated Analysis for BCI Signals & EEG</p>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR: CONFIGURATION & PIPELINE MANAGEMENT
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Pipeline Configuration")
    
    st.markdown("---")
    st.markdown("### Step 1: Initialize Assistant Brain")
    
    raw_api_key = st.text_input(
        "Enter Google API Key", 
        type="password", 
        placeholder="AIza...",
        help="Get this from Google AI Studio. Stored in session only."
    )
    st.caption("Press Enter to apply")
    
    is_key_valid = False
    if raw_api_key:
        if raw_api_key.startswith("AIza") and len(raw_api_key) > 30:
            os.environ["GOOGLE_API_KEY"] = raw_api_key
            st.success("Assistant initialized! Key format valid.")
            is_key_valid = True
        else:
            st.error("Invalid API Key format. It should start with 'AIza' and be ~39 characters long.")
    
    st.markdown("---")
    st.markdown("### Step 2: Ingest Signal Data")
    uploaded_file = st.file_uploader(
        "Load experimental CSV data", 
        type=["csv"], 
        help="Expected columns: Subject_ID, Condition, Alpha_Power, Beta_Power..."
    )

    OUTPUT_DIR = os.path.abspath("./ui_graphs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    st.markdown("---")
    st.markdown("### 🛠️ Quick Analysis Actions")
    q_btn_1 = st.button("Generate Alpha/Beta Boxplot", help="Analysis like 'theta_power_boxplot.png'")
    q_btn_2 = st.button("Summary Descriptive Stats", help="Instantly calculate average signal powers")
    
    if st.button("🗑️ Clear Pipeline History"):
        if "messages" in st.session_state:
            st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------
# MAIN WORKSPACE: TELEMETRY & ANALYSIS
# ---------------------------------------------------------
if uploaded_file and is_key_valid:
    df = pd.read_csv(uploaded_file)
    
    st.markdown("### 📡 Data Telemetry")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Subject Count", df['Subject_ID'].nunique() if 'Subject_ID' in df.columns else "N/A")
    col2.metric("Features Detected", df.shape[1])
    col3.metric("Signal Rows (Total)", f"{df.shape[0]:,}")
    
    if 'Alpha_Power' in df.columns:
        col4.metric("Avg Alpha Power", f"{df['Alpha_Power'].mean():.2f} µV")
    else:
        col4.metric("Avg Alpha Power", "N/A")

    st.divider()

    # ---------------------------------------------------------
    # THE CLEAN IMPORT (No duplication!)
    # ---------------------------------------------------------
    agent = get_neural_agent(df, OUTPUT_DIR)

    tab1, tab2 = st.tabs(["💬 Assistant Chat", "📋 Data Preview"])
    
    with tab2:
        st.markdown("### Descriptive Statistics Summary")
        st.dataframe(df.describe().T, use_container_width=True)
        st.markdown("### Head of Dataset")
        st.dataframe(df.head(100), use_container_width=True)

    with tab1:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        input_container = st.container()
        query_input = st.chat_input("Request analysis or visualization of the current dataset...")
        
        if q_btn_1: query_input = "Generate a boxplot showing the distribution of Alpha_Power and Beta_Power for each Condition. Save the plot."
        if q_btn_2: query_input = "Show me the mean Alpha_Power and Beta_Power grouped by Subject_ID."

        if query_input:
            st.session_state.messages.append({"role": "user", "content": query_input})
            with st.chat_message("user"):
                st.markdown(query_input)

            with st.chat_message("assistant"):
                with st.spinner("Assistant thinking... analyzing signals..."):
                    for f in glob.glob(f"{OUTPUT_DIR}/*.png"):
                        os.remove(f)

                    try:
                        answer = agent.invoke(query_input)
                        response_text = answer['output']
                        
                        # Clean up the prefix if it leaks
                        clean_text = response_text.replace("Final Answer:", "").strip()
                        
                        st.markdown(clean_text)
                        st.session_state.messages.append({"role": "assistant", "content": clean_text})

                        generated_images = glob.glob(f"{OUTPUT_DIR}/*.png")
                        if generated_images:
                            for img_path in generated_images:
                                st.image(img_path, use_container_width=True)
                                with open(img_path, "rb") as file:
                                    st.download_button(label="💾 Download Plot", data=file, file_name=os.path.basename(img_path), mime="image/png")
                    except Exception as e:
                        error_msg = f"Analysis Error: {e}"
                        if "PERMISSION_DENIED" in error_msg:
                            st.error("API Key Authentication Failed. Please check your key in the sidebar.")
                        else:
                            st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

elif uploaded_file and not is_key_valid:
    st.info("👈 Data ingested successfully. Please provide a valid API Key in the sidebar to unlock the analysis tools.")
elif not uploaded_file:
    st.success("✨ Welcome. Your Signal Pipeline is initialized. Please use the sidebar to load your dataset.")