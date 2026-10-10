from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import datetime
from db_conn import db
import logging
from sql__A import ask

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SQL Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    success: bool
    execution_time: Optional[float] = None

class TableResponse(BaseModel):
    tables: list
    total_tables: int

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    database_connected: bool

@app.get("/")
async def home():
    return "Welcome, SQL Agent API is running with database connection"

@app.get('/health', response_model=HealthResponse)
async def health_check():
    try:
        db.run("SELECT 1")
        db_connected = True
    except Exception as e:
        logger.error(f"Database Connection error: {e}")
        db_connected = False

    return HealthResponse(
        status="healthy" if db_connected else "unhealthy",
        timestamp=datetime.datetime.now().isoformat(),
        database_connected=db_connected
    )

@app.post('/query', response_model=QueryResponse)
async def query_db(request: QueryRequest):
    start_time = datetime.datetime.now()
    try:
        result = ask(request.question)
        answer = result
        execution_time = (datetime.datetime.now() - start_time).total_seconds()

        logger.info(f"Query executed successfully in {execution_time:.2f}s")
        logger.info(f"Answer: {answer[:200]}...")

        return QueryResponse(
            answer=answer,
            success=True,
            execution_time=execution_time
        )
    
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        execution_time = (datetime.datetime.now() - start_time).total_seconds()

        return QueryResponse(
            answer=f"Sorry, I encountered an error: {str(e)}",
            success=False,
            execution_time=execution_time
        )

@app.get('/tables', response_model=TableResponse)
async def get_tables():
    try:
        from sql__A import toolkit

        list_tables_tool = next(
            tool for tool in toolkit.get_tools()
            if tool.name == "sql_db_list_tables"
        )
        result = list_tables_tool.invoke({})
        tables = result.split(',') if isinstance(result, str) else result

        return TableResponse(
            tables=tables,
            total_tables=len(tables)
        )
    
    except Exception as e:
        logger.error(f"Error fetching tables: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching tables: {str(e)}")

@app.get('/table-schema/{table_name}')
async def get_table_schema(table_name: str):
    """Get schema columns for a specific table"""
    try:
        from sql__A import toolkit
        import re

        get_schema_tool = next(
            tool for tool in toolkit.get_tools()
            if tool.name == "sql_db_schema"
        )
        result = get_schema_tool.invoke(table_name)

        logger.info(f"Raw schema for {table_name}:\n{result}")

        columns = []

        # For SQLite, try using PRAGMA table_info for accurate metadata
        try:
            pragma_result = db.run(f"PRAGMA table_info({table_name})")
            logger.info(f"PRAGMA result: {pragma_result}")

            # PRAGMA returns: cid|name|type|notnull|dflt_value|pk
            if pragma_result:
                lines = pragma_result.strip().split('\n')
                for line in lines:
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 6:
                        col_name = parts[1]
                        col_type = parts[2]
                        is_pk = int(parts[5]) > 0 if parts[5] else False

                        columns.append({
                            "name": col_name,
                            "type": col_type,
                            "primary_key": is_pk
                        })

                logger.info(f"Columns from PRAGMA: {columns}")
                if columns:
                    return {"table_name": table_name, "columns": columns, "raw_schema": result}
        except Exception as e:
            logger.warning(f"PRAGMA table_info failed: {e}, falling back to text parsing")

        # Fallback: Parse the schema text
        lines = result.split('\n')

        # First pass: Find all PRIMARY KEY constraints and column-level PKs
        pk_columns = set()

        for line in lines:
            # Check for table-level PRIMARY KEY constraint: PRIMARY KEY (col1, col2, ...)
            if 'PRIMARY KEY' in line.upper() and '(' in line:
                match = re.search(r'PRIMARY\s+KEY\s*\(\s*([^)]+)\s*\)', line, re.IGNORECASE)
                if match:
                    pk_list = match.group(1)
                    # Extract column names
                    for col in pk_list.split(','):
                        col_name = col.strip().strip('"').strip("'")
                        pk_columns.add(col_name)
                        logger.info(f"Found PK column from table-level constraint: {col_name}")

        # Second pass: Process columns
        in_create_table = False
        for line in lines:
            if 'CREATE TABLE' in line.upper():
                in_create_table = True
                continue

            if in_create_table:
                # Stop at closing parenthesis
                if line.strip() == ')' or line.strip() == ');':
                    break

                line = line.strip()
                if not line or line.startswith('--'):
                    continue

                # Parse column definition: "name" TYPE [constraints]
                match = re.match(r'"([^"]+)"\s+(\w+)(?:\([^)]*\))?\s*(.*)', line)

                if match:
                    col_name = match.group(1)
                    col_type = match.group(2)
                    constraints_str = match.group(3).rstrip(',').strip().upper() if match.group(3) else ""

                    # Check both column-level and table-level PK constraints
                    is_pk = 'PRIMARY KEY' in constraints_str or col_name in pk_columns

                    logger.info(f"Column: {col_name}, Type: {col_type}, PK: {is_pk}, Constraints: {constraints_str}, In PK Set: {col_name in pk_columns}")

                    columns.append({
                        "name": col_name,
                        "type": col_type,
                        "primary_key": is_pk
                    })

        return {"table_name": table_name, "columns": columns, "raw_schema": result}

    except Exception as e:
        logger.error(f"Error fetching schema for {table_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching schema: {str(e)}")