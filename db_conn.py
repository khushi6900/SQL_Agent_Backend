import os
from langchain_community.utilities import SQLDatabase

# Get absolute path to the database file relative to script directory
db_path = os.path.abspath("Chinook.db")
print("Connecting to:", db_path)

if not os.path.exists(db_path):
    raise FileNotFoundError(db_path)
db = SQLDatabase.from_uri(f"sqlite:///{db_path}?mode=ro")

# Verify available tables
print("Tables in DB:", db.get_usable_table_names())