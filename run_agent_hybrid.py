import json
import click
import dspy
import pandas as pd
from tqdm import tqdm
from agent.graph_hybrid import build_graph, setup_dspy

@click.command()
@click.option('--batch', required=True, help='Path to input JSONL file')
@click.option('--out', required=True, help='Path to output JSONL file')
def main(batch, out):
    # 1. Setup DSPy
    setup_dspy()
    
    # 2. Load Data
    questions = []
    with open(batch, 'r') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    
    # 3. Initialize Graph
    app = build_graph()
    
    results = []
    
    # 4. Run Inference
    print(f"Processing {len(questions)} questions...")
    for q in tqdm(questions):
        initial_state = {
            "question": q["question"],
            "format_hint": q.get("format_hint", "str"),
            "repair_count": 0,
            "retrieved_docs": [],
            "constraints": {},
            "sql_query": "",
            "sql_results": [],
            "sql_error": None,
            "citations": []
        }
        
        try:
            # Using invoke to run the graph
            final_state = app.invoke(initial_state)
            
            # Extract outputs
            output = {
                "id": q["id"],
                "final_answer": final_state.get("final_answer"),
                "sql": final_state.get("sql_query", ""),
                "confidence": 0.8 if not final_state.get("sql_error") else 0.2, # Simple heuristic
                "explanation": "Generated based on hybrid analysis of docs and DB.",
                "citations": final_state.get("citations", [])
            }
            results.append(output)
            
        except Exception as e:
            print(f"Error processing {q['id']}: {e}")
            results.append({
                "id": q["id"],
                "final_answer": None,
                "sql": "",
                "confidence": 0.0,
                "explanation": f"Error: {str(e)}",
                "citations": []
            })
            
    # 5. Write Output
    with open(out, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
            
    print(f"Done! Results written to {out}")

if __name__ == '__main__':
    main()

