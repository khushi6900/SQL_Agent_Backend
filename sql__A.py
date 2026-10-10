from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from db_conn import db
import os

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ===================================
# 1. LLM
# ===================================

model = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

# =================================== 
# 2. Tools Setup
# ===================================
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()
# print(tools)

# """
# Tool 1: sql_db_query - Input to this tool is a detailed and correct SQL query, output is a result from the database. If the query is not correct, an error message will be returned. If an error is returned, rewrite the query, check the query, and try again. If you encounter an issue with Unknown column 'xxxx' in 'field list', use sql_db_schema to query the correct table fields.
# Tool 2: sql_db_schema - Input to this tool is a comma-separated list of tables, output is the schema and sample rows for those tables. Be sure that the tables actually exist by calling sql_db_list_tables first! Example Input: table1, table2, table3
# Tool 3: sql_db_list_tables - Input is an empty string, output is a comma-separated list of tables in the database.
# Tool 4: sql_db_query_checker - Use this tool to double check if your query is correct before executing it. Always use this tool before executing a query with sql_db_query!
# """

# =================================== 
# 3. Prompt Templates
# ===================================
system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
""".format(
        dialect=db.dialect,
        top_k=5,
    )

# ===================================
# 4. Agent (LangGraph)
# ===================================

agent = create_react_agent(
    model,
    tools,                      # from toolkit.get_tools()
    prompt=system_prompt,       
    checkpointer=MemorySaver(), # optional: chat memory per thread
)

def ask(question, thread_id="default"):
    result = agent.invoke(
        {"messages": [("user", question)]},
        config={"configurable": {"thread_id": thread_id}}
    )
    # print("Result:", result)
    msg = result["messages"][-1]
    return msg.text if hasattr(msg, "text") and not callable(msg.text) else msg.content

# =================================== 
# 5. Testing function
# ===================================
def test_agent(questions):
    """Test the SQL agent with various questions"""
    for question in questions:
        print(f"\n{'='*60}")
        print(f"TESTING: {question}")
        print(f"{'='*60}")
        
        # Invoke the agent
        result = ask(question)
        
        print(f"\nFINAL ANSWER: {result}")
        print(f"{'='*60}")

# =================================== 
# 6. Execution
# ===================================
if __name__ == "__main__":
    test_questions = [
        # "Total number of artists in the db?",
        # "Give me all unique track names", 
        # "Show me 5 customers from London",
        "How many invoices are there?"
    ]
    
    test_agent(test_questions)