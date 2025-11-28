import sys
import os
import dspy

# Ensure we can import from the agent package
sys.path.append(os.getcwd())

from agent.dspy_signatures import TextToSQL
from agent.tools.sqlite_tool import SQLiteTool

# Define the metric: SQL must execute successfully
def sql_metric(example, pred, trace=None):
    tool = SQLiteTool()
    sql = pred.sql_query
    # Cleanup
    sql = sql.replace("```sql", "").replace("```", "").strip()
    results, cols, error = tool.execute_sql(sql)
    return error is None

def optimize_sql_module():
    # 1. Setup
    # Updated for latest DSPy: Use dspy.LM with ollama_chat prefix
    lm = dspy.LM(model="ollama_chat/phi3.5:3.8b-mini-instruct-q4_K_M", api_base="http://localhost:11434", api_key="")
    dspy.settings.configure(lm=lm)
    
    tool = SQLiteTool()
    schema = tool.get_schema(["Orders", "Order Details", "Products"])
    
    # 2. Create Training Data (Few-Shot)
    train_examples = [
        dspy.Example(
            question="How many products are there?",
            schema=schema,
            constraints="None",
            sql_query="SELECT COUNT(*) FROM Products;"
        ).with_inputs("question", "schema", "constraints"),
        
        dspy.Example(
            question="What is the total revenue from Order 10248?",
            schema=schema,
            constraints="Revenue = UnitPrice * Quantity * (1-Discount)",
            sql_query="SELECT SUM(UnitPrice * Quantity * (1 - Discount)) FROM \"Order Details\" WHERE OrderID = 10248;"
        ).with_inputs("question", "schema", "constraints"),
        
        dspy.Example(
            question="List all products in CategoryID 1.",
            schema=schema,
            constraints="None",
            sql_query="SELECT ProductName FROM Products WHERE CategoryID = 1;"
        ).with_inputs("question", "schema", "constraints"),

        dspy.Example(
            question="Total sales in Q1 1997?",
            schema=schema,
            constraints="date_range_start='1997-01-01', date_range_end='1997-03-31'",
            sql_query="SELECT SUM(UnitPrice * Quantity * (1 - Discount)) FROM \"Order Details\" JOIN Orders ON \"Order Details\".OrderID = Orders.OrderID WHERE OrderDate BETWEEN '1997-01-01' AND '1997-03-31';"
        ).with_inputs("question", "schema", "constraints")
    ]
    
    # 3. Optimize
    print("Starting optimization with BootstrapFewShot...")
    from dspy.teleprompt import BootstrapFewShot
    
    teleprompter = BootstrapFewShot(metric=sql_metric, max_bootstrapped_demos=2)
    compiled_sql = teleprompter.compile(TextToSQL(), trainset=train_examples)
    
    # 4. Save
    output_path = os.path.join("agent", "optimized_sql_module.json")
    compiled_sql.save(output_path)
    print(f"Optimization complete! Saved to {output_path}")

if __name__ == "__main__":
    optimize_sql_module()
