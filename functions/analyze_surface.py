import os
from langchain_openai import ChatOpenAI
from models import BugReport, ErrorX

# -------------------- LLM --------------------

llm = ChatOpenAI(          # ← now only runs when the function is called
        model="gpt-4o-mini",
        openai_api_key=os.environ.get("OPENAI_KEY"),
        temperature=0.3,
    )

_structured_llm = llm.with_structured_output(BugReport)

# -------------------- PROMPT --------------------
PROMPT_TEMPLATE = '''
You are an expert software system analyzer.

Your task is NOT to generate bugs or code.

Your goal is to analyze the repository structure and identify:

1. Valid executable or interactive files ONLY (ignore README, configs, static files)
2. Entry points where data enters or logic is triggered
3. Possible bug categories applicable at those entry points

----------------------------------

Project:


{tree_structure}
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

'''



async def analyze_surface_node(state: ErrorX) -> dict:
    tree_structure = state.get("tree_structure", "")   # 👈 set by build_tree_node
    logs = state.get("logs", [])
    errors = state.get("errors", [])

    if not tree_structure:
        return {
            **state,
            "bug_report": {},
            "errors": errors + ["analyze_surface: tree_structure is empty or missing"],
        }

    try:
        prompt = PROMPT_TEMPLATE.format(tree_structure=tree_structure)
        response: BugReport = await _structured_llm.ainvoke(prompt)

        total_files = sum(
            len(layer.items)
            for layer in [response.backend, response.frontend, response.database]
        )

        return {
            **state,
            "bug_report": response.dict(),
            "logs": logs + [f"analyze_surface: mapped {total_files} files across 3 layers"],
        }

    except Exception as e:
        return {
            **state,
            "bug_report": {},
            "errors": errors + [f"analyze_surface: {str(e)}"],
        }