import dspy
import sqlite3
import os
from typing import Annotated, TypedDict, List, Dict, Any, Optional, Union
from langgraph.graph import StateGraph, END
from agent.dspy_signatures import Router, Planner, TextToSQL, Synthesizer
from agent.rag.retrieval import SimpleRetriever
from agent.tools.sqlite_tool import SQLiteTool

# --- Setup DSPy (User should configure this before running) ---
def setup_dspy():
    # Updated for latest DSPy: Use dspy.LM with ollama_chat prefix
    lm = dspy.LM(model="ollama_chat/phi3.5:3.8b-mini-instruct-q4_K_M", api_base="http://localhost:11434", api_key="")
    dspy.settings.configure(lm=lm)

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
    router = Router()
    pred = router(question=state["question"])
    # Fallback if classification is weird
    cls = pred.classification.lower().strip()
    if "hybrid" in cls: return {"classification": "hybrid"}
    if "sql" in cls: return {"classification": "sql"}
    return {"classification": "rag"}

def retriever_node(state: AgentState):
    retriever = SimpleRetriever() # Reloads index, inefficient but stateless safe
    docs = retriever.search(state["question"], k=3)
    return {"retrieved_docs": docs}

def planner_node(state: AgentState):
    planner = Planner()
    context_str = "\n\n".join([d['content'] for d in state["retrieved_docs"]])
    try:
        pred = planner(question=state["question"], context=context_str)
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
        
    return {"constraints": constraints}

def sql_generator_node(state: AgentState):
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
    
    # Clean SQL (remove markdown code blocks if present)
    raw_sql = pred.sql_query.strip()
    clean_sql = raw_sql.replace("```sql", "").replace("```", "").strip()
    
    return {"sql_query": clean_sql, "schema": schema}

def executor_node(state: AgentState):
    tool = SQLiteTool()
    results, cols, error = tool.execute_sql(state["sql_query"])
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
    synthesizer = Synthesizer()
    
    context_str = "\n\n".join([d['content'] for d in state.get("retrieved_docs", [])])
    sql_info = f"SQL: {state.get('sql_query')}\nResults: {state.get('sql_results')}"
    
    pred = synthesizer(
        question=state["question"],
        sql_query=state.get("sql_query", ""),
        sql_result=str(state.get("sql_results", [])),
        retrieved_context=context_str,
        format_hint=state["format_hint"]
    )
    
    # Parse citations and answer
    return {
        "final_answer": pred.final_answer,
        "citations": pred.citations
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

