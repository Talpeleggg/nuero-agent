import streamlit as st
import pandas as pd
import os
import glob
from agent import get_neural_agent

# 1. Page Configuration (Centered and focused)
st.set_page_config(page_title="BCI Data Pipeline", page_icon="🧠", layout="centered")

# --- Header & Brain Aesthetics ---
# Using a high-quality unsplash brain/network image
st.image("https://images.unsplash.com/photo-1559757175-5700dde675bc?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80", use_column_width=True)
st.markdown("<h1 style='text-align: center; color: #2E86C1; margin-top: -20px;'>🧠 BCI Signal Analysis Pipeline</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #5D6D7E; font-size: 1.2rem;'>Automated Neurotechnology & EEG Data Agent</p>", unsafe_allow_html=True)
st.divider()

# --- Section 1: Authentication ---
st.markdown("### 🔑 1. System Authentication")
st.info("Please enter your Google API Key below to activate the neural agent.")
api_key = st.text_input("Google API Key", type="password", placeholder="AIzaSy...", help="Press Enter to apply")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    st.success("Authentication Successful! Pipeline unlocked.")
    st.divider()

    # --- Section 2: Data Ingestion ---
    st.markdown("### 📂 2. Data Ingestion")
    uploaded_file = st.file_uploader("Upload your EEG/BCI Dataset (CSV format)", type=["csv"])
    
    OUTPUT_DIR = os.path.abspath("./ui_graphs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # Initialize the fixed agent!
        agent = get_neural_agent(df, OUTPUT_DIR)
        
        # --- Section 3: Telemetry ---
        st.markdown("### 📡 3. Dataset Telemetry")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Samples (Rows)", f"{df.shape[0]:,}")
        col2.metric("Features (Cols)", df.shape[1])
        col3.metric("Missing Values", df.isna().sum().sum())
        col4.metric("Memory", f"{df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
        
        with st.expander("🔍 View Descriptive Statistics"):
            st.dataframe(df.describe(), use_container_width=True)
        st.divider()

        # --- Section 4: Quick Diagnostics ---
        st.markdown("### ⚡ 4. Quick Diagnostics")
        colA, colB, colC = st.columns(3)
        prompt_1 = colA.button("📊 Alpha vs Beta Power", use_container_width=True)
        prompt_2 = colB.button("📈 Data Distribution Boxplot", use_container_width=True)
        prompt_3 = colC.button("🧠 Highest Signal Value", use_container_width=True)
        st.divider()

        # --- Section 5: Chat Interface ---
        col_chat, col_clear = st.columns([0.8, 0.2])
        with col_chat:
            st.markdown("### 💬 5. Neural Agent Chat")
        with col_clear:
            if st.button("🗑️ Clear History"):
                if "messages" in st.session_state:
                    st.session_state.messages = []
                st.rerun()

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_prompt = st.chat_input("Ask the agent to analyze the neural signals...")
        
        if prompt_1: user_prompt = "Create a bar chart comparing average Alpha and Beta power across conditions. Save it as a png."
        if prompt_2: user_prompt = "Create a boxplot showing the distribution of all continuous variables. Save it as a png."
        if prompt_3: user_prompt = "Which row or condition has the highest recorded signal value? Give me the exact numbers."

        if user_prompt:
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.chat_message("assistant"):
                with st.spinner("Processing neural data..."):
                    for f in glob.glob(f"{OUTPUT_DIR}/*.png"):
                        os.remove(f)
                    
                    try:
                        answer = agent.invoke(user_prompt)
                        response_text = answer['output']
                        
                        # Remove the explicit "Final Answer:" text if it leaked into the UI
                        clean_text = response_text.replace("Final Answer:", "").strip()
                        
                        st.markdown(clean_text)
                        st.session_state.messages.append({"role": "assistant", "content": clean_text})

                        generated_images = glob.glob(f"{OUTPUT_DIR}/*.png")
                        if generated_images:
                            for img_path in generated_images:
                                st.image(img_path)
                                with open(img_path, "rb") as file:
                                    st.download_button("💾 Download Plot", data=file, file_name=os.path.basename(img_path), mime="image/png")
                    except Exception as e:
                        error_msg = f"Pipeline Error: {e}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})