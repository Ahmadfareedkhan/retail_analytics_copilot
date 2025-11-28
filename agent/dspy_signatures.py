import dspy
from typing import List, Optional

# --- Router ---
class RouterSignature(dspy.Signature):
    """Classify the user question into one of three categories: 'rag', 'sql', or 'hybrid'.
    - 'rag': Questions about policies, dates, catalogs, or static text info.
    - 'sql': Questions requiring aggregations, exact numbers from DB, or specific order details.
    - 'hybrid': Questions needing both defined terms/dates (from docs) AND database queries.
    """
    question = dspy.InputField(desc="The user's retail analytics question")
    classification = dspy.OutputField(desc="One of: 'rag', 'sql', 'hybrid'")

class Router(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(RouterSignature)
    
    def forward(self, question):
        return self.prog(question=question)

# --- Planner (Extract Constraints) ---
class PlannerSignature(dspy.Signature):
    """Analyze the question and retrieved docs to extract query constraints."""
    question = dspy.InputField()
    context = dspy.InputField(desc="Relevant chunks from documentation")
    
    date_range_start = dspy.OutputField(desc="Start date YYYY-MM-DD or None")
    date_range_end = dspy.OutputField(desc="End date YYYY-MM-DD or None")
    kpi_formula = dspy.OutputField(desc="Relevant KPI formula text or None")
    entities = dspy.OutputField(desc="List of relevant entities (products, categories, customers)")

class Planner(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(PlannerSignature)
        
    def forward(self, question, context):
        return self.prog(question=question, context=context)

# --- NL to SQL ---
class TextToSQLSignature(dspy.Signature):
    """Generate a SQLite query based on the question, schema, and constraints.
    
    CRITICAL RULES:
    - Table names: Orders, "Order Details" (with quotes!), Products, Categories, Customers
    - Date filtering: WHERE OrderDate BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'
    - NEVER use made-up keywords like BETWEDIR, BETWEWHEN
    - For JOINs: JOIN "Order Details" od ON Orders.OrderID = od.OrderID
    - Return ONLY valid SQLite syntax starting with SELECT
    """
    question = dspy.InputField()
    db_schema = dspy.InputField(desc="SQLite CREATE TABLE statements")
    constraints = dspy.InputField(desc="Specific constraints (dates, formulas) from docs")
    
    sql_query = dspy.OutputField(desc="Valid SQLite query with proper table quoting")
    explanation = dspy.OutputField(desc="Brief explanation of the logic")

class TextToSQL(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(TextToSQLSignature)
        
    def forward(self, question, db_schema, constraints):
        return self.prog(question=question, db_schema=db_schema, constraints=constraints)

# --- Synthesizer ---
class SynthesizerSignature(dspy.Signature):
    """Synthesize a final answer matching the format hint.
    - If format_hint is 'int', return only the number.
    - If format_hint is 'float', return the number.
    - If format_hint is JSON structure, return valid JSON.
    - Cite sources used (table names, doc chunk IDs).
    """
    question = dspy.InputField()
    sql_query = dspy.InputField(desc="Executed SQL query (if any)")
    sql_result = dspy.InputField(desc="Result rows from SQL execution")
    retrieved_context = dspy.InputField(desc="Text chunks from documentation")
    format_hint = dspy.InputField(desc="Expected output format (e.g., 'int', 'float', 'json')")
    
    final_answer = dspy.OutputField(desc="The typed answer matching format_hint exactly")
    citations = dspy.OutputField(desc="List of strings: table names and doc chunk IDs used")

class Synthesizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(SynthesizerSignature)
        
    def forward(self, question, sql_query, sql_result, retrieved_context, format_hint):
        return self.prog(
            question=question,
            sql_query=sql_query,
            sql_result=sql_result,
            retrieved_context=retrieved_context,
            format_hint=format_hint
        )
