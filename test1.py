import os
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI

# -----------------------------
# Load environment
# -----------------------------
load_dotenv()

# -----------------------------
# Schema Definitions
# -----------------------------

class BugSurface(BaseModel):
    file: str = Field(...)
    entry_points: List[str] = Field(...)
    bug_types: List[str] = Field(...)

class LayerSurface(BaseModel):
    items: List[BugSurface] = Field(...)

class BugReport(BaseModel):
    backend: LayerSurface = Field(...)
    frontend: LayerSurface = Field(...)
    database: LayerSurface = Field(...)

# -----------------------------
# Initialize LLM
# -----------------------------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_KEY"),
    temperature=0
)

structured_llm = llm.with_structured_output(BugReport)

# -----------------------------
# Prompt
# -----------------------------

PROMPT = """
You are an expert software system analyzer.

Your task is NOT to generate bugs or code.

Your goal is to analyze the repository structure and identify:

1. Valid executable or interactive files ONLY (ignore README, configs, static files)
2. Entry points where data enters or logic is triggered
3. Possible bug categories applicable at those entry points

----------------------------------

Project:

ai-stock-platform/
├── backend/app/api/routes/stocks.py
├── backend/app/api/routes/users.py
├── backend/app/services/auth_service.py
├── backend/app/services/stock_service.py
├── frontend/src/pages/Dashboard.jsx
├── frontend/src/pages/Login.jsx
├── frontend/src/services/api.js
├── database/models.sql
├── database/migrations/

----------------------------------

Instructions:

Backend:
- Focus on API routes, services
- Entry points = endpoints, request handlers, auth flows

Frontend:
- Entry points = user interactions, API calls, state updates

Database:
- Entry points = queries, schema operations, migrations

----------------------------------

Bug categories:

Frontend:
- async
- state
- null
- event

Backend:
- api_contract
- auth
- logic
- validation

Database:
- injection
- transaction
- index
- query

----------------------------------

Output rules:

- DO NOT include description
- DO NOT include code
- DO NOT assume functions
- DO NOT guess entry points
- ONLY include valid files with real execution role
- IGNORE non-relevant files

Return ONLY structured output.
"""

# -----------------------------
# Run Function
# -----------------------------

def run():
    response = structured_llm.invoke(PROMPT)

    print("\n=== ⚡ ENTRY POINT ANALYSIS ===")

    for layer_name, layer in response.dict().items():
        print(f"\n--- {layer_name.upper()} ---")
        
        for item in layer["items"]:
            print(f"\n📂 {item['file']}")
            print(f"  Entry Points: {item['entry_points']}")
            print(f"  Bug Types: {item['bug_types']}")

    return response


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    run()