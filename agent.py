import os
import google.auth
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

def get_neural_agent(df, output_dir):
    """
    Creates the LangChain Pandas agent using Enterprise Authentication standards.
    """
    # ENTERPRISE AUTHENTICATION STRATEGY
    # 1. Try to load local .env for development fallback
    load_dotenv()
    
    # 2. Check for Application Default Credentials (Kubernetes Workload Identity)
    try:
        credentials, project = google.auth.default()
        print("System: Using Kubernetes/GCP Workload Identity.")
    except google.auth.exceptions.DefaultCredentialsError:
        print("System: ADC not found. Falling back to local .env API key.")
        if not os.getenv("GOOGLE_API_KEY"):
            raise EnvironmentError("No K8s Identity and no GOOGLE_API_KEY found.")

    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    # THE ELITE GUARDRAILS
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
        You MUST begin your final response with the exact prefix: "Final Answer: ".
        """
        
    agent = create_pandas_dataframe_agent(
        llm, 
        df, 
        verbose=True, 
        allow_dangerous_code=True,
        prefix=instructions, 
        handle_parsing_errors=True
    )
    
    return agent