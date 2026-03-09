import streamlit as st
import pandas as pd
import os
import glob
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# 1. UI Configuration
st.set_page_config(page_title="Neuro-Agent UI", layout="wide")
st.title("🧠 Neuroscience AI Data Assistant")
st.markdown("Upload your EEG dataset and ask the AI agent to analyze or plot the data.")

# 2. Sidebar for API Key (Better security practice than hardcoding)
api_key = st.sidebar.text_input("Enter your Google API Key:", type="password")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

# 3. Cloud-Ready Directory Management
OUTPUT_DIR = os.path.abspath("./ui_graphs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 4. File Uploader UI
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

# Only run the agent if a file is uploaded AND the API key is provided
if uploaded_file is not None and api_key:
    
    # Read the data and show a preview in the UI
    df = pd.read_csv(uploaded_file)
    st.write("### 📊 Data Preview")
    st.dataframe(df.head())

    # Initialize the Brain
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    instructions = f"""
    You are an expert Neuroscience Data Analyst. 
    You are working with a pandas dataframe containing EEG brain wave data.
    Strict Rules:
    1. If asked to create a graph or plot, use matplotlib or seaborn.
    2. ALWAYS save the plot as a '.png' file exactly in this directory: '{OUTPUT_DIR}'.
    3. Do NOT use plt.show(). Tell the user the plot was saved successfully.
    """

    agent = create_pandas_dataframe_agent(
        llm, df, verbose=True, allow_dangerous_code=True,
        prefix=instructions, handle_parsing_errors=True
    )

    st.write("---")
    st.write("### 🤖 Ask the Agent")
    
    # 5. Question Box and Button
    question = st.text_input("What would you like to analyze or plot?")
    
    if st.button("Run Analysis") and question:
        # Show a loading spinner while the agent thinks
        with st.spinner("Agent is writing code and analyzing data..."):
            
            # Housekeeping: Delete old graphs from the folder before running
            for f in glob.glob(f"{OUTPUT_DIR}/*.png"):
                os.remove(f)

            try:
                # Run the agent
                answer = agent.invoke(question)
                
                # Show the text answer
                st.success("Analysis Complete!")
                st.info(answer['output'])

                # 6. The Magic File Detection & Download
                # Check if the agent created any new PNG files in the folder
                generated_images = glob.glob(f"{OUTPUT_DIR}/*.png")
                
                if generated_images:
                    for img_path in generated_images:
                        # Display the image in the UI
                        st.image(img_path, caption="Agent Generated Plot")
                        
                        # Create the download button for the local PC
                        with open(img_path, "rb") as file:
                            st.download_button(
                                label="💾 Download Graph to PC",
                                data=file,
                                file_name=os.path.basename(img_path),
                                mime="image/png"
                            )
            except Exception as e:
                st.error(f"An error occurred: {e}")

elif not api_key:
    st.warning("👈 Please enter your Google API Key in the sidebar to begin.")
