import streamlit as st
import os
import glob
from agent import get_neural_agent
from data_ingestion import process_neuro_data

st.set_page_config(page_title="NeuroData Pipeline", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .hero-banner { background: linear-gradient(135deg, #101015 0%, #202025 100%); padding: 2.5rem; border-radius: 12px; margin-bottom: 2rem; border: 1px solid #303035; }
    .hero-title { color: #FFFFFF; font-size: 2.5rem !important; font-weight: 700 !important; margin-bottom: 0.5rem !important; }
    .hero-subtitle { color: #A0A0A5; font-size: 1.1rem; margin-bottom: 0px !important; }
    .stChatMessage { border-radius: 10px; margin-bottom: 10px; }
    .stChatMessage.user { background-color: #262730; }
    .stChatMessage.assistant { background-color: #101015; border: 1px solid #202025; }
    div.stMetric { background-color: #1A1A20; border-radius: 10px; padding: 15px; border: 1px solid #202025; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">🧠 NeuroData Pipeline</h1>
        <p class="hero-subtitle">Enterprise BCI Analysis with Parquet Compression & Auto-Denoising</p>
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Pipeline Configuration")
    
    st.markdown("### 🔐 Security Status")
    st.success("Identity Managed via K8s/ADC")
    
    st.markdown("---")
    st.markdown("### 📡 1. Ingest Data")
    uploaded_file = st.file_uploader("Load CSV or EDF", type=["csv", "edf"])
    
    st.markdown("### 🎛️ 2. Pre-Processing")
    apply_denoise = st.checkbox("Apply 1-40Hz Bandpass Filter (EDF only)", help="Removes 50/60Hz line noise and low-frequency drift.")
    
    OUTPUT_DIR = os.path.abspath("./ui_graphs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if st.button("🗑️ Clear Pipeline History"):
        if "messages" in st.session_state:
            st.session_state.messages = []
        st.rerun()

if uploaded_file:
    with st.spinner("ETL Engine: Processing and Compressing to Parquet..."):
        try:
            # Route to our new module!
            df, parquet_path = process_neuro_data(uploaded_file, apply_denoise)
            agent = get_neural_agent(df, OUTPUT_DIR)
            st.session_state['data_loaded'] = True
        except Exception as e:
            st.error(f"Ingestion Error: {e}")
            st.stop()

    if st.session_state.get('data_loaded'):
        st.markdown("### 📊 Optimized Data Telemetry")
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Features (Channels)", df.shape[1])
        col2.metric("Signal Rows", f"{df.shape[0]:,}")
        col3.metric("Missing Values", df.isna().sum().sum())
        # Show the power of Parquet vs raw Memory
        parquet_size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
        col4.metric("Parquet Storage Size", f"{parquet_size_mb:.2f} MB")

        st.divider()

        tab1, tab2 = st.tabs(["💬 Assistant Chat", "📋 Data Preview"])
        
        with tab2:
            st.dataframe(df.head(100), use_container_width=True)

        with tab1:
            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            query_input = st.chat_input("Request analysis on the compressed dataset...")

            if query_input:
                st.session_state.messages.append({"role": "user", "content": query_input})
                with st.chat_message("user"):
                    st.markdown(query_input)

                with st.chat_message("assistant"):
                    with st.spinner("Analyzing signals..."):
                        for f in glob.glob(f"{OUTPUT_DIR}/*.png"):
                            os.remove(f)

                        try:
                            answer = agent.invoke(query_input)
                            clean_text = answer['output'].replace("Final Answer:", "").strip()
                            st.markdown(clean_text)
                            st.session_state.messages.append({"role": "assistant", "content": clean_text})

                            generated_images = glob.glob(f"{OUTPUT_DIR}/*.png")
                            if generated_images:
                                for img_path in generated_images:
                                    st.image(img_path, use_container_width=True)
                                    with open(img_path, "rb") as file:
                                        st.download_button("💾 Download Plot", data=file, file_name=os.path.basename(img_path), mime="image/png")
                        except Exception as e:
                            st.error(f"Analysis Error: {e}")
elif not uploaded_file:
    st.info("👈 System Ready. Upload an EDF or CSV file in the sidebar to begin the ETL pipeline.")