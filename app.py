import streamlit as st
import pandas as pd
import os
import glob

# Import the function from agent.py
from agent import get_neural_agent

# 1. Page Configuration
st.set_page_config(
    page_title="BCI Data Pipeline",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI polish
st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; color: #2E86C1; font-weight: bold; text-align: center; margin-bottom: 0px;}
    .sub-header {font-size: 1.2rem; color: #5D6D7E; text-align: center; margin-bottom: 30px;}
    </style>
    <div class="main-header">🧠 BCI Signal Analysis Agent</div>
    <div class="sub-header">Automated Pipeline for Neurotechnology & EEG Datasets</div>
    """, unsafe_allow_html=True)

# 2. Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Pipeline Config")
    api_key = st.text_input("Google API Key:", type="password", help="Securely stored in session only.")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        st.success("API Key Verified!")
    
    st.markdown("---")
    st.markdown("### 🛠️ Quick Diagnostics")
    # Quick action buttons to save you typing time
    prompt_1 = st.button("📊 Plot Alpha vs Beta Power")
    prompt_2 = st.button("📈 Show Data Distribution")
    prompt_3 = st.button("🧠 Find Highest Signal Value")
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        if "messages" in st.session_state:
            st.session_state.messages = []
        st.rerun()

# 3. Output Directory setup (Cloud/Container ready)
OUTPUT_DIR = os.path.abspath("./ui_graphs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 4. Main Interface - File Upload
uploaded_file = st.file_uploader("Upload your EEG/BCI Dataset (CSV)", type=["csv"])

if uploaded_file and api_key:
    # Read the data
    df = pd.read_csv(uploaded_file)
    
    # ---------------------------------------------------------
    # CREATE THE AGENT DYNAMICALLY HERE
    # ---------------------------------------------------------
    agent = get_neural_agent(df, OUTPUT_DIR)
    
    # --- Top Section: Dataset Telemetry ---
    st.markdown("### 📡 Dataset Telemetry")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Samples (Rows)", f"{df.shape[0]:,}")
    col2.metric("Features (Columns)", df.shape[1])
    col3.metric("Missing Values", df.isna().sum().sum())
    col4.metric("Memory Footprint", f"{df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
    
    # Expanders for clean UI without taking up the whole screen
    with st.expander("🔍 View Raw Dataset & Descriptive Statistics"):
        tab_raw, tab_stats = st.tabs(["Raw Data", "Statistical Summary"])
        with tab_raw:
            st.dataframe(df.head(100), use_container_width=True)
        with tab_stats:
            st.dataframe(df.describe(), use_container_width=True)

    st.divider()

    # --- Chat Interface ---
    st.markdown("### 💬 Neural Agent Interface")
    
    # Initialize session state for chat memory
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Determine the prompt (either from chat input or the quick buttons)
    user_prompt = st.chat_input("Ask the agent to analyze the neural signals...")
    
    if prompt_1: user_prompt = "Create a bar chart comparing average Alpha and Beta power across conditions. Save it as a png."
    if prompt_2: user_prompt = "Create a boxplot showing the distribution of all continuous variables. Save it as a png."
    if prompt_3: user_prompt = "Which row or condition has the highest recorded signal value? Give me the exact numbers."

    if user_prompt:
        # 1. Add user message
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # 2. Get Agent response
        with st.chat_message("assistant"):
            with st.spinner("Processing neural data..."):
                # Clean up old graphs so we don't display the wrong one
                for f in glob.glob(f"{OUTPUT_DIR}/*.png"):
                    os.remove(f)

                try:
                    answer = agent.invoke(user_prompt)
                    response_text = answer['output']
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})

                    # Detect and display new plots
                    generated_images = glob.glob(f"{OUTPUT_DIR}/*.png")
                    if generated_images:
                        for img_path in generated_images:
                            st.image(img_path)
                            with open(img_path, "rb") as file:
                                st.download_button(
                                    label="💾 Download Plot", 
                                    data=file, 
                                    file_name=os.path.basename(img_path), 
                                    mime="image/png"
                                )
                except Exception as e:
                    error_msg = f"Pipeline Error: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

elif not api_key:
    st.info("👈 Please securely enter your Google API Key in the sidebar to initialize the pipeline.")