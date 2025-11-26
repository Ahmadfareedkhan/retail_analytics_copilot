import dspy
import os
import sys

# Add project root to path to allow imports
sys.path.append(os.getcwd())

from agent.dspy_signatures import TextToSQL
from agent.tools.sqlite_tool import SQLiteTool

# Define the metric: SQL must execute successfully
def sql_metric(example, pred, trace=None):
    tool = SQLiteTool()
    sql = pred.sql_query
    # Cleanup
    if sql:
        sql = sql.replace("```sql", "").replace("```", "").strip()
    else:
        return False
        
    results, cols, error = tool.execute_sql(sql)
    return error is None

def optimize_sql_module():
    # 1. Setup
    lm = dspy.OllamaLocal(model="phi3.5:3.8b-mini-instruct-q4_K_M", max_tokens=1000)
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
        ).with_inputs("question", "schema", "constraints")
    ]
    
    # 3. Optimize
    print("Starting optimization with BootstrapFewShot...")
    from dspy.teleprompt import BootstrapFewShot
    
    teleprompter = BootstrapFewShot(metric=sql_metric, max_bootstrapped_demos=2)
    compiled_sql = teleprompter.compile(TextToSQL(), trainset=train_examples)
    
    # 4. Save
    if not os.path.exists("agent"):
        os.makedirs("agent")
    
    compiled_sql.save("agent/optimized_sql_module.json")
    print("Optimization complete! Saved to agent/optimized_sql_module.json")

if __name__ == "__main__":
    optimize_sql_module()
