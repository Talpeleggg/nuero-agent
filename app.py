import streamlit as st
import pandas as pd
import os
import glob
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# 1. High-End Page Configuration (Inspired by modern data apps)
st.set_page_config(
    page_title="NeuroData Pipeline",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CUSTOM STYLING (Replacing deprecated elements and adding aesthetics)
# ---------------------------------------------------------
st.markdown("""
    <style>
    /* Professional Header Bar with integrated brain art */
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

    /* Clean Chat Interface */
    .stChatMessage { border-radius: 10px; margin-bottom: 10px; }
    .stChatMessage.user { background-color: #262730; }
    .stChatMessage.assistant { background-color: #101015; border: 1px solid #202025; }

    /* Info Cards */
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
    
    # Secure API Input
    st.markdown("---")
    st.markdown("### Step 1: Initialize Assistant Brain")
    api_key = st.text_input(
        "Enter Google API Key", 
        type="password", 
        help="Get this from Google AI Studio. Stored in session only."
    )
    
    # Securely set environment variable
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        st.success("Assistant initialized!")
    
    # Data Ingestion
    st.markdown("---")
    st.markdown("### Step 2: Ingest Signal Data")
    uploaded_file = st.file_uploader(
        "Load experimental CSV data", 
        type=["csv"], 
        help="Expected columns: Subject_ID, Condition, Alpha_Power, Beta_Power..."
    )

    # Output management
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
if uploaded_file and api_key:
    # Read the data instantly
    df = pd.read_csv(uploaded_file)
    
    # ---------------------------------------------------------
    # UI SECTION A: DATA TELEMETRY (Inspired by dashboards)
    # ---------------------------------------------------------
    st.markdown("### 📡 Data Telemetry")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Subject Count", df['Subject_ID'].nunique())
    col2.metric("Features Detected", df.shape[1])
    col3.metric("Signal Rows (Total)", f"{df.shape[0]:,}")
    
    # Find Avg Alpha Power if column exists
    if 'Alpha_Power' in df.columns:
        avg_alpha = df['Alpha_Power'].mean()
        col4.metric("Avg Alpha Power", f"{avg_alpha:.2f} µV")
    else:
        col4.metric("Avg Alpha Power", "N/A")

    st.divider()

    # Initialize LLM & Agent
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
    
    instructions = f"""
    You are an expert Neuroscience Data Analyst specializing in BCI signal analysis.
    You are working with a pandas dataframe. Strict Rules:
    1. If asked to create a graph, generate it using matplotlib or seaborn.
    2. ALWAYS save the plot as a '.png' file exactly in this directory: '{OUTPUT_DIR}'.
    3. State clearly that the plot was saved. Never use `plt.show()`.
    4. Format your analytical findings professionally.
    """
    
    agent = create_pandas_dataframe_agent(
        llm, 
        df, 
        verbose=True, 
        allow_dangerous_code=True,
        prefix=instructions, 
        handle_parsing_errors=True
    )

    # ---------------------------------------------------------
    # UI SECTION B: ANALYSIS WORKSPACE (Tabs)
    # ---------------------------------------------------------
    tab1, tab2 = st.tabs(["💬 Assistant Chat", "📋 Data Preview"])
    
    with tab2:
        st.markdown("### Descriptive Statistics Summary")
        st.dataframe(df.describe().T, use_container_width=True)
        st.markdown("### Head of Dataset")
        st.dataframe(df.head(100), use_container_width=True)

    with tab1:
        # Chat interface management
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat Input Area (The prompt text box)
        input_container = st.container()
        
        # Accept quick button prompts or user input
        query_input = st.chat_input("Request analysis or visualization of the current dataset...")
        
        # Overwrite user input if quick button was pressed
        if q_btn_1: query_input = "Generate a boxplot showing the distribution of Alpha_Power and Beta_Power for each Condition. Save the plot."
        if q_btn_2: query_input = "Show me the mean Alpha_Power and Beta_Power grouped by Subject_ID."

        # Handle Query Execution
        if query_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": query_input})
            with st.chat_message("user"):
                st.markdown(query_input)

            # Get Agent Response
            with st.chat_message("assistant"):
                with st.spinner("Assistant thinking... analyzing signals..."):
                    # Clear old graphs to ensure clean display
                    for f in glob.glob(f"{OUTPUT_DIR}/*.png"):
                        os.remove(f)

                    try:
                        answer = agent.invoke(query_input)
                        response_text = answer['output']
                        st.markdown(response_text)
                        st.session_state.messages.append({"role": "assistant", "content": response_text})

                        # Detect and show generated images
                        generated_images = glob.glob(f"{OUTPUT_DIR}/*.png")
                        if generated_images:
                            for img_path in generated_images:
                                st.image(img_path, use_container_width=True)
                                # Add download button for professionalism
                                with open(img_path, "rb") as file:
                                    st.download_button(label="💾 Download Plot", data=file, file_name=os.path.basename(img_path), mime="image/png")
                    except Exception as e:
                        error_msg = f"Analysis Error: {e}"
                        # Securely handle common API key errors
                        if "PERMISSION_DENIED" in error_msg:
                            st.error("API Key Authentication Failed (Permission Denied). Please check your key in the sidebar.")
                        elif "reported as leaked" in error_msg:
                            st.error("API Key compromised. Please revoke and provide a fresh key.")
                        else:
                            st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

elif uploaded_file and not api_key:
    # Handle the "Ingested, but missing config" state aestheticallly
    st.info("👈 Data ingested successfully. Please configure and initialize the Assistant Assistant brain in the sidebar to begin analysis.")
elif not uploaded_file:
    # Empty State (Professionally organized)
    st.success("✨ Welcome. Your Neuroscience Signal Pipeline is initialized. Please use the sidebar to load your dataset.")