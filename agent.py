import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# 1. Paste your Google AI Studio API key here
os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY_HERE"

# 2. Dynamic Data Loader (The Best Practice)
# We tell Pandas to read the dataset from our external CSV file
file_path = "brain_data.csv"
print(f"Attempting to load dataset from: {file_path}...\n")

try:
    df = pd.read_csv(file_path)
    print(f"Success! Loaded {len(df)} rows of data into the agent's memory.\n")
except FileNotFoundError:
    print(f"Error: Could not find '{file_path}'. Please make sure it's in the same folder.")
    exit()

# 3. Define the LLM (The "Brain" of the agent) using Gemini
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 4. Define the Guardrails (The Agent's Persona)
# This replaces the need for predefined scenarios!
instructions = """
You are an expert Neuroscience Data Analyst. 
You are working with a pandas dataframe containing EEG brain wave data.
Strict Rules:
1. If the user asks a question unrelated to the data, politely decline to answer.
2. If the user asks you to create a graph or plot, write Python code to generate it using matplotlib or seaborn.
3. ALWAYS save the plot as a '.png' file using plt.savefig() and do NOT use plt.show(). Tell the user the name of the saved file.
"""

# 5. Create the Agent with the Guardrails
agent = create_pandas_dataframe_agent(
    llm, 
    df, 
    verbose=True, 
    allow_dangerous_code=True,
    prefix=instructions # Injecting our guardrails here!
)

# 6. The Interactive Research Loop
print("=" * 40)
print("  Neuro-Agent Activated  ")
print("  Type 'exit' to close ")
print("=" * 40 + "\n")

while True:
    scientist_question = input("\nWhat would you like to analyze or plot? \n> ")

    if scientist_question.lower() in ['exit', 'quit']:
        print("Shutting down the agent. Goodbye!")
        break

    print("\nAgent is working...")
    try:
        answer = agent.invoke(scientist_question)
        print(f"\nFinal Answer: {answer['output']}\n")
    except Exception as e:
        print(f"\nAn error occurred: {e}\n")
