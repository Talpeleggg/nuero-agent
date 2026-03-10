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
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)   
    
    # ---------------------------------------------------------
    # THE ELITE GUARDRAILS (System Prompt)
    # ---------------------------------------------------------
    instructions = f"""
        You are an elite Computational Neuroscientist and Lead Data Engineer. 
        You specialize in analyzing Brain-Computer Interface (BCI) signals and EEG data.
        
        METHODOLOGY:
        1. Check for missing values (NaN) before providing statistical summaries.
        2. Group by experimental conditions when calculating metrics (e.g., average frequency power).
        
        VISUALIZATION RULES:
        1. Generate production-quality graphs using 'matplotlib' or 'seaborn'.
        2. ALWAYS save the plot as a '.png' file exactly in this directory: '{output_dir}'.
        3. NEVER use `plt.show()`.
        
        CRITICAL FORMATTING RULE:
        When you are ready to provide the final output to the user, you MUST begin your final response with the exact prefix: "Final Answer: ". If you do not use this prefix, the system will crash.
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
    
    # 1. Load secrets and configs from the .env file
    load_dotenv() 
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("CRITICAL ERROR: GOOGLE_API_KEY is missing. Please add it to your .env file.")
        exit(1)

    # 2. Robust path routing for outputs
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.abspath("./saved_graphs"))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 3. Dynamic Data Loading (No more hardcoding!)
    # It looks for TEST_DATA_PATH in .env, and defaults to "brain_data.csv" if it accidentally got deleted
    test_file_path = os.getenv("TEST_DATA_PATH", "brain_data.csv")
    
    try:
        print(f"System: Attempting to load test dataset from '{test_file_path}'...")
        df = pd.read_csv(test_file_path)
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
        print(f"Error: Could not find '{test_file_path}'. Please check your .env file or ensure the file exists.")