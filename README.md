# SQL Agent Backend

FastAPI backend for the SQL Agent application.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the backend directory:
```env
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

4. Place the `Chinook.db` database file in the backend folder.

## Running the Backend

```bash
cd backend
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

- `GET /` - Health check
- `GET /health` - Database health status
- `POST /query` - Execute SQL query
- `GET /tables` - List all tables
- `GET /table-schema/{table_name}` - Get table schema