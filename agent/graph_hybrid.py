import dspy
import sqlite3
import os
import time
from typing import Annotated, TypedDict, List, Dict, Any, Optional, Union
from langgraph.graph import StateGraph, END
from agent.dspy_signatures import Router, Planner, TextToSQL, Synthesizer
from agent.rag.retrieval import SimpleRetriever
from agent.tools.sqlite_tool import SQLiteTool

# --- Setup DSPy (User should configure this before running) ---
def setup_dspy():
    # Disable caching to prevent sticky hallucinations
    os.environ["DSP_CACHEBOOL"] = "False"
    
    # Updated for latest DSPy: Use dspy.LM with ollama_chat prefix
    lm = dspy.LM(model="ollama_chat/phi3.5:3.8b-mini-instruct-q4_K_M", api_base="http://localhost:11434", api_key="")
    dspy.settings.configure(lm=lm)

# Global retriever cache
_RETRIEVER_INSTANCE = None
def get_retriever():
    global _RETRIEVER_INSTANCE
    if _RETRIEVER_INSTANCE is None:
        _RETRIEVER_INSTANCE = SimpleRetriever()
    return _RETRIEVER_INSTANCE

# --- State Definition ---
class AgentState(TypedDict):
    question: str
    format_hint: str
    
    # Internal State
    classification: Optional[str]
    retrieved_docs: List[Dict]  # [{'id':..., 'content':..., 'score':...}]
    constraints: Dict[str, Any] # From planner
    
    schema: str
    sql_query: Optional[str]
    sql_results: Optional[List[Dict]]
    sql_columns: Optional[List[str]]
    sql_error: Optional[str]
    
    final_answer: Any
    citations: List[str]
    
    repair_count: int
    repair_feedback: Optional[str]

# --- Nodes ---

def router_node(state: AgentState):
    start = time.time()
    print(f"DEBUG: Starting Router Node")
    router = Router()
    try:
        pred = router(question=state["question"])
        # Fallback if classification is weird
        cls = pred.classification.lower().strip()
        print(f"DEBUG: Router finished in {time.time() - start:.2f}s. Classification: {cls}")
        if "hybrid" in cls: return {"classification": "hybrid"}
        if "sql" in cls: return {"classification": "sql"}
        return {"classification": "rag"}
    except Exception as e:
        print(f"Router error: {e}, defaulting to hybrid")
        return {"classification": "hybrid"}

def retriever_node(state: AgentState):
    start = time.time()
    print(f"DEBUG: Starting Retriever Node")
    retriever = get_retriever() # Use cached instance
    docs = retriever.search(state["question"], k=1)
    print(f"DEBUG: Retriever finished in {time.time() - start:.2f}s")
    return {"retrieved_docs": docs}

def planner_node(state: AgentState):
    start = time.time()
    print(f"DEBUG: Starting Planner Node")
    planner = Planner()
    context_str = "\n\n".join([d['content'] for d in state["retrieved_docs"]])
    try:
        # Force bypass cache by adding a timestamp to the input (hacky but effective for debugging)
        # or just rely on the fact that we are debugging.
        pred = planner(question=state["question"], context=context_str)
        
        print(f"DEBUG: Planner Output: {pred}")
        
        constraints = {
            "date_range_start": getattr(pred, 'date_range_start', None),
            "date_range_end": getattr(pred, 'date_range_end', None),
            "kpi_formula": getattr(pred, 'kpi_formula', None),
            "entities": getattr(pred, 'entities', [])
        }
    except Exception as e:
        # Fallback for parsing errors
        print(f"Planner parsing error: {e}, using default constraints")
        constraints = {
            "date_range_start": None, "date_range_end": None, 
            "kpi_formula": None, "entities": []
        }
    print(f"DEBUG: Planner finished in {time.time() - start:.2f}s")
    return {"constraints": constraints}

def sql_generator_node(state: AgentState):
    start = time.time()
    print(f"DEBUG: Starting SQL Generator Node")
    tool = SQLiteTool()
    schema = tool.get_schema() # Get full schema or subset
    
    # Prepare constraints string
    constraints_str = str(state.get("constraints", "None"))
    if state.get("repair_feedback"):
        constraints_str += f"\nPREVIOUS ERROR: {state['repair_feedback']}. FIX THIS."

    # Load optimized module if available
    generator = TextToSQL()
    opt_path = os.path.join(os.getcwd(), "agent", "optimized_sql_module.json")
    if os.path.exists(opt_path):
        try:
            generator.load(opt_path)
            print(f"Loaded optimized SQL module from {opt_path}")
        except Exception as e:
            print(f"Warning: Could not load optimized module: {e}")

    pred = generator(
        question=state["question"],
        schema=schema,
        constraints=constraints_str
    )
    
    print(f"DEBUG: SQL Generator Output: {pred}")
    
    # Clean SQL (remove markdown code blocks if present)
    if hasattr(pred, 'sql_query'):
        raw_sql = pred.sql_query.strip()
        clean_sql = raw_sql.replace("```sql", "").replace("```", "").strip()
    else:
        clean_sql = ""
        print("DEBUG: No sql_query field in prediction!")
    
    print(f"DEBUG: SQL Generator finished in {time.time() - start:.2f}s")
    return {"sql_query": clean_sql, "schema": schema}

def executor_node(state: AgentState):
    start = time.time()
    print(f"DEBUG: Starting Executor Node")
    tool = SQLiteTool()
    results, cols, error = tool.execute_sql(state["sql_query"])
    print(f"DEBUG: Executor finished in {time.time() - start:.2f}s")
    return {
        "sql_results": results,
        "sql_columns": cols,
        "sql_error": error
    }

def repair_check_node(state: AgentState):
    """Logic to decide if we need to repair SQL."""
    if state["sql_error"] and state["repair_count"] < 2:
        return "repair"
    return "synthesize"

def repair_node(state: AgentState):
    return {
        "repair_count": state["repair_count"] + 1,
        "repair_feedback": state["sql_error"] or "Invalid format or empty result"
    }

def synthesizer_node(state: AgentState):
    start = time.time()
    print(f"DEBUG: Starting Synthesizer Node")
    synthesizer = Synthesizer()
    
    context_str = "\n\n".join([d['content'] for d in state.get("retrieved_docs", [])])
    
    # Truncate results to avoid context overflow
    results = state.get("sql_results", [])
    if results and len(results) > 20:
        results_str = str(results[:20]) + f"\n... ({len(results)-20} more rows omitted)"
    else:
        results_str = str(results)
        
    sql_info = f"SQL: {state.get('sql_query')}\nResults: {results_str}"
    
    try:
        pred = synthesizer(
            question=state["question"],
            sql_query=state.get("sql_query", ""),
            sql_result=str(state.get("sql_results", [])),
            retrieved_context=context_str,
            format_hint=state["format_hint"]
        )
        final_answer = pred.final_answer
        citations = pred.citations
    except Exception as e:
        print(f"Synthesizer error: {e}")
        # Fallback: try to extract answer from error message or return basic error
        final_answer = f"Error synthesizing answer: {str(e)}"
        citations = []

    # Parse citations and answer
    print(f"DEBUG: Synthesizer finished in {time.time() - start:.2f}s")
    return {
        "final_answer": final_answer,
        "citations": citations
    }

# --- Graph Construction ---

def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("router", router_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("sql_generator", sql_generator_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("repair", repair_node)
    workflow.add_node("synthesizer", synthesizer_node)
    
    workflow.set_entry_point("router")
    
    # Router logic
    workflow.add_conditional_edges(
        "router",
        lambda x: x["classification"],
        {
            "rag": "retriever",
            "sql": "sql_generator",
            "hybrid": "retriever"
        }
    )
    
    # Retriever Logic
    workflow.add_conditional_edges(
        "retriever",
        lambda x: x["classification"],
        {
            "rag": "synthesizer", # Skip SQL for pure RAG
            "hybrid": "planner"
        }
    )
    
    workflow.add_edge("planner", "sql_generator")
    workflow.add_edge("sql_generator", "executor")
    
    # Executor -> Repair or Synthesize
    workflow.add_conditional_edges(
        "executor",
        repair_check_node,
        {
            "repair": "repair",
            "synthesize": "synthesizer"
        }
    )
    
    workflow.add_edge("repair", "sql_generator") # Loop back to try SQL again
    workflow.add_edge("synthesizer", END)
    
    return workflow.compile()

