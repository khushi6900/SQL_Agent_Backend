from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits import create_sql_agent
from dotenv import load_dotenv
from db_conn import db
import os

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ===================================
# 1. LLM
# ===================================
# Using Qwen2.5-7B-Instruct - widely available on HF Inference API, good at SQL
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN
)

model = ChatHuggingFace(llm=llm)

# =================================== 
# 2. Tools Setup
# ===================================
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

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
database.""".format(
    dialect=db.dialect,
    top_k=5,
)

# =================================== 
# 4. Create Agent
# ===================================
agent = create_sql_agent(
    llm=model,
    toolkit=toolkit,
    verbose=True,
    agent_type="openai-tools",
)# from typing import Literal
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from langchain_community.agent_toolkits import SQLDatabaseToolkit
# from langchain.agents import create_agent
# from langchain_core.messages import AIMessage, HumanMessage
# from langgraph.graph import END, START, MessagesState, StateGraph
# from langgraph.prebuilt import ToolNode
# from dotenv import load_dotenv
# from db_conn import db
# import os

# load_dotenv()

# HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ===================================
# 1. LLM
# ===================================
# Using Qwen2.5-7B-Instruct - widely available on HF Inference API, good at SQL
# llm = HuggingFaceEndpoint(
#     repo_id="openai/gpt-oss-20b",
#     task="text-generation",
#     huggingfacehub_api_token=HF_TOKEN
# )

# model = ChatHuggingFace(llm=llm)

# # =================================== 
# # 2. Tools Setup
# # ===================================
# toolkit = SQLDatabaseToolkit(db=db, llm=model)
# tools = toolkit.get_tools()
# # print(tools)

# """
# Tool 1: sql_db_query - Input to this tool is a detailed and correct SQL query, output is a result from the database. If the query is not correct, an error message will be returned. If an error is returned, rewrite the query, check the query, and try again. If you encounter an issue with Unknown column 'xxxx' in 'field list', use sql_db_schema to query the correct table fields.
# Tool 2: sql_db_schema - Input to this tool is a comma-separated list of tables, output is the schema and sample rows for those tables. Be sure that the tables actually exist by calling sql_db_list_tables first! Example Input: table1, table2, table3
# Tool 3: sql_db_list_tables - Input is an empty string, output is a comma-separated list of tables in the database.
# Tool 4: sql_db_query_checker - Use this tool to double check if your query is correct before executing it. Always use this tool before executing a query with sql_db_query!
# """

# # Get the specific tools we need
# list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
# get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
# run_query_tool = next(tool for tool in tools if tool.name == "sql_db_query")

# # Create tool nodes
# get_schema_node = ToolNode([get_schema_tool], name="get_schema")
# run_query_node = ToolNode([run_query_tool], name="run_query")

# # =================================== 
# # 3. Prompt Templates
# # ===================================
# generate_query_system_prompt = """
# You are an agent designed to interact with a SQL database.
# Given an input question, create a syntactically correct {dialect} query to run,
# then look at the results of the query and return the answer. Unless the user
# specifies a specific number of examples they wish to obtain, always limit your
# query to at most {top_k} results.

# You can order the results by a relevant column to return the most interesting
# examples in the database. Never query for all the columns from a specific table,
# only ask for the relevant columns given the question.

# DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
# """.format(
#     dialect=db.dialect,
#     top_k=5,
# )

# check_query_system_prompt = """
# You are a SQL expert with a strong attention to detail.
# Double check the {dialect} query for common mistakes, including:
# - Using NOT IN with NULL values
# - Using UNION when UNION ALL should have been used
# - Using BETWEEN for exclusive ranges
# - Data type mismatch in predicates
# - Properly quoting identifiers
# - Using the correct number of arguments for functions
# - Casting to the correct data type
# - Using the proper columns for joins

# If there are any of the above mistakes, rewrite the query. If there are no mistakes,
# just reproduce the original query.

# Also each time you generate a query, return the generated query along with the final response.

# You will call the appropriate tool to execute the query after running this check.
# """.format(dialect=db.dialect)

# # =================================== 
# # 4. Nodes function
# # ===================================
# def list_tables(state: MessagesState):
#     """Start by listing all available tables in the database"""
#     tool_call = {
#         "name": "sql_db_list_tables",
#         "args": {},
#         "id": "abc123",
#         "type": "tool_call",
#     }
#     tool_call_message = AIMessage(content="", tool_calls=[tool_call])

#     # Execute the tool
#     tool_message = list_tables_tool.invoke(tool_call)
    
#     # Create a response message with the table information
#     response = AIMessage(content=f"Available tables: {tool_message}")

#     return {"messages": [tool_call_message, tool_message, response]}


# def call_get_schema(state: MessagesState):
#     """Force the model to decide which tables to get schemas for"""
#     try:
#         # llm_with_tools = model.bind_tools([get_schema_tool], tool_choice="any")
#         llm_with_tools = model.bind_tools([get_schema_tool])
#         response = llm_with_tools.invoke(state["messages"])
#         return {"messages": [response]}
#     except StopIteration as e:
#         # Handle huggingface_hub provider resolution failure
#         error_msg = AIMessage(content=f"Error: Failed to connect to Hugging Face endpoint. Please check your API key and model availability. Details: {e}")
#         return {"messages": [error_msg]}


# def generate_query(state: MessagesState):
#     """Generate SQL query based on the question and schema"""
#     try:
#         system_message = {
#             "role": "system",
#             "content": generate_query_system_prompt,
#         }

#         # Allow model to respond naturally or call tools
#         llm_with_tools = model.bind_tools([run_query_tool])
#         response = llm_with_tools.invoke([system_message] + state["messages"])
#         return {"messages": [response]}
#     except StopIteration as e:
#         error_msg = AIMessage(content=f"Error: Failed to generate query. Hugging Face endpoint issue: {e}")
#         return {"messages": [error_msg]}


# def check_query(state: MessagesState):
#     """Validate and correct the SQL query before execution"""
#     try:
#         system_message = {
#             "role": "system",
#             "content": check_query_system_prompt,
#         }

#         # Get the query from the last tool call
#         tool_call = state["messages"][-1].tool_calls[0]
#         user_message = {"role": "user", "content": tool_call["args"]["query"]}

#         # Force the model to call the query tool
#         # llm_with_tools = model.bind_tools([run_query_tool], tool_choice="any")
#         llm_with_tools = model.bind_tools([run_query_tool])
#         response = llm_with_tools.invoke([system_message, user_message])

#         # Preserve the message ID for consistency
#         response.id = state["messages"][-1].id

#         return {"messages": [response]}
#     except StopIteration as e:
#         error_msg = AIMessage(content=f"Error: Failed to check query. Hugging Face endpoint issue: {e}")
#         return {"messages": [error_msg]}


# def should_continue(state: MessagesState) -> Literal[END, "check_query"]:
#     """Determine whether to continue or end the workflow"""
#     messages = state["messages"]
#     last_message = messages[-1]
    
#     # If no tool calls, we have the final answer
#     if not last_message.tool_calls:
#         return END
#     else:
#         # If there are tool calls, check the query before execution
#         return "check_query"

# # =================================== 
# # 5. Graph
# # ===================================
# builder = StateGraph(MessagesState)

# # Add all nodes
# builder.add_node("list_tables", list_tables)
# builder.add_node("call_get_schema", call_get_schema)
# builder.add_node("get_schema", get_schema_node)
# builder.add_node("generate_query", generate_query)
# builder.add_node("check_query", check_query)
# builder.add_node("run_query", run_query_node)

# # Add edges
# builder.add_edge(START, "list_tables")
# builder.add_edge("list_tables", "call_get_schema")
# builder.add_edge("call_get_schema", "get_schema")
# builder.add_edge("get_schema", "generate_query")
# builder.add_conditional_edges("generate_query", should_continue)
# builder.add_edge("check_query", "run_query")
# builder.add_edge("run_query", "generate_query")

# # Compile the agent
# agent = builder.compile()

# # =================================== 
# # 6. Testing function
# # ===================================
# def test_agent(questions):
#     """Test the SQL agent with various questions"""
#     for question in questions:
#         print(f"\n{'='*60}")
#         print(f"TESTING: {question}")
#         print(f"{'='*60}")
        
#         # Invoke the agent
#         result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        
#         # Extract and display the final answer
#         final_message = result["messages"][-1]
        
#         if hasattr(final_message, 'content'):
#             print(f"FINAL ANSWER: {final_message.content}")
#         else:
#             print(f"No final answer found")
#             print(f"Last message: {final_message}")

# # =================================== 
# # 7. Execution
# # ===================================
# if __name__ == "__main__":
#     # Test cases
#     test_questions = [
#         "Total number of artists in the db?",
#         # "Give me all unique track names", 
#         # "Show me 5 customers from London",
#         # "How many invoices are there?"
#     ]
#     test_agent(test_questions)





from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits import create_sql_agent
from dotenv import load_dotenv
from db_conn import db
import os

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ===================================
# 1. LLM
# ===================================
# Using Qwen2.5-7B-Instruct - widely available on HF Inference API, good at SQL
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN
)

model = ChatHuggingFace(llm=llm)

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

agent = create_sql_agent(
    llm=model,
    db=db,
    # toolkit=toolkit,
    system_prompt=system_prompt,
    verbose=True,
    agent_type="openai-tools",
    allow_dangerous_code=True
)


# =================================== 
# 6. Testing function
# ===================================
def test_agent(agent, questions):
    """Test the SQL agent with various questions"""
    for question in questions:
        print(f"\n{'='*60}")
        print(f"TESTING: {question}")
        print(f"{'='*60}")
        
        # Invoke the agent
        result = agent.invoke({"input": question})
        
        print(f"\nFINAL ANSWER: {result['output']}")
        print(f"{'='*60}")

# =================================== 
# 7. Execution
# ===================================
if __name__ == "__main__":
    test_questions = [
        # "Total number of artists in the db?",
        # "Give me all unique track names", 
        "Show me 5 customers from London",
        "How many invoices are there?"
    ]
    
    test_agent(agent, test_questions)