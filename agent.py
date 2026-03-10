import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

def get_neural_agent(df, output_dir):
    """
    Creates and returns a production-ready LangChain Pandas agent 
    with advanced guardrails for neurotechnology datasets.
    """
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    # ---------------------------------------------------------
    # THE ELITE GUARDRAILS (System Prompt)
    # ---------------------------------------------------------
    instructions = f"""
    You are an elite Computational Neuroscientist and Lead Data Engineer. 
    You specialize in analyzing Brain-Computer Interface (BCI) signals, EEG data, and complex neurotechnology datasets.
    
    Your goal is to assist researchers by analyzing the provided pandas dataframe.
    
    METHODOLOGY & DATA RIGOR:
    1. Always check for missing values (NaN) or potential anomalies (like motion artifacts or extreme voltage spikes) before providing statistical summaries.
    2. When calculating metrics (e.g., average frequency power, latency), ensure you group by the relevant experimental conditions if they exist in the data.
    3. Explain your findings in clear, academic language suitable for a lab report or research paper.
    
    VISUALIZATION RULES:
    1. If asked to plot data, generate production-quality graphs using 'matplotlib' or 'seaborn'.
    2. Ensure all graphs have clear titles, axis labels (with units like Hz, µV, or ms if applicable), and legends.
    3. CRITICAL: You MUST save EVERY plot as a '.png' file exactly in this directory: '{output_dir}'.
    4. NEVER use `plt.show()`. Your environment cannot render it.
    5. After saving, state clearly: "I have generated the plot and saved it to [insert filename]."
    
    SECURITY & SCOPE:
    1. You are strictly confined to analyzing the provided dataframe. 
    2. If the user asks you to perform tasks outside of data analysis (e.g., writing web scrapers, executing OS system commands, or answering general trivia), politely refuse and guide them back to the dataset.
    """
    
    # Create the Agent
    agent = create_pandas_dataframe_agent(
        llm, 
        df, 
        verbose=True, 
        allow_dangerous_code=True,
        prefix=instructions, 
        handle_parsing_errors=True
    )
    
    return agent

# ==========================================
# LOCAL CLI TESTING BLOCK
# ==========================================
if __name__ == "__main__":
    import pandas as pd
    
    # 1. Load secrets from the .env file securely!
    load_dotenv() 
    
    # Check if the key was loaded successfully
    if not os.getenv("GOOGLE_API_KEY"):
        print("CRITICAL ERROR: GOOGLE_API_KEY is missing. Please add it to your .env file.")
        exit(1)

    # 2. Robust path routing
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.abspath("./saved_graphs"))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        df = pd.read_csv("brain_data.csv")
        cli_agent = get_neural_agent(df, OUTPUT_DIR)
        
        print("=" * 50)
        print(" 🧠 Elite Neural-Agent CLI Activated ")
        print("  Type 'exit' to close ")
        print("=" * 50 + "\n")

        while True:
            scientist_question = input("\nWhat is your analysis request? \n> ")
            if scientist_question.lower() in ['exit', 'quit']:
                print("Terminating session. Goodbye!")
                break
            
            try:
                answer = cli_agent.invoke(scientist_question)
                print(f"\n[Agent]: {answer['output']}\n")
            except Exception as e:
                print(f"\n[System Error]: {e}\n")
                
    except FileNotFoundError:
        print("Error: Could not find 'brain_data.csv' for CLI testing.")