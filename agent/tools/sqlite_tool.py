import sqlite3
from typing import List, Dict, Any, Tuple, Optional

class SQLiteTool:
    def __init__(self, db_path: str = "data/northwind.sqlite"):
        self.db_path = db_path

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def list_tables(self) -> List[str]:
        """List all user-defined tables in the database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tables

    def get_schema(self, table_names: Optional[List[str]] = None) -> str:
        """Get the CREATE TABLE statements for specified tables or all tables."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if not table_names:
            table_names = self.list_tables()
            
        schema_str = []
        for table in table_names:
            cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,))
            res = cursor.fetchone()
            if res:
                schema_str.append(res[0] + ";")
        
        conn.close()
        return "\n\n".join(schema_str)

    def execute_sql(self, query: str) -> Tuple[List[Dict[str, Any]], List[str], Optional[str]]:
        """
        Execute a SQL query.
        Returns: (results as list of dicts, column names, error message)
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(query)
            if query.strip().lower().startswith("select"):
                results = [dict(row) for row in cursor.fetchall()]
                columns = [description[0] for description in cursor.description] if cursor.description else []
                conn.close()
                return results, columns, None
            else:
                conn.commit()
                conn.close()
                return [], [], None
        except Exception as e:
            conn.close()
            return [], [], str(e)

