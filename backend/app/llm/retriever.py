from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List

from psycopg2.extras import RealDictCursor
from app.utils import db
from app.utils.db import DatabaseManager
from app.config import load_config

class RehabDbRetriever(BaseRetriever):
    """A custom retriever that fetches a user's progress from our database."""
    user_id: str

    def invoke(self, query: str) -> List[Document]:
        """Custom invoke method to bypass complex runtime setup if needed."""
        # Simple invocation without run_manager logic for the chain
        return self._get_relevant_documents(query)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None # 💡 run_manager เป็น optional
    ) -> List[Document]:
        
        db_manager = DatabaseManager()
        context_str = ""
        
        # --- Logic to fetch exercise list ---
        if "ท่า" in query or "exercise" in query or "สรุป" in query: # 💡 ตรวจสอบ 'สรุป' เพื่อดึง context เพิ่มเติม
            exercises = list(load_config()['exercises'].keys()) 
            if exercises:
                context_str += f"Available exercises: {', '.join(exercises)}.\n"

        # --- Logic to fetch user progress ---
        # ... (SQL and database fetching logic remains the same) ...
        sql = """
        SELECT exercise_id, is_success, metric_errors, rep_timestamp
        FROM rep_history
        WHERE user_id = %s
        ORDER BY rep_timestamp DESC
        LIMIT 5;
        """
        
        try:
            with db_manager.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                 cursor.execute(sql, (self.user_id,))
                 recent_progress = cursor.fetchall()
        except Exception as e:
            print(f"Error fetching progress for RAG: {e}")
            recent_progress = []

        if recent_progress:
            progress_summary = "\nRecent User Performance (last 5 reps):\n"
            for row in recent_progress:
                progress_summary += f"- {row['rep_timestamp'].strftime('%Y-%m-%d %H:%M')}: {row['exercise_id']}, Success: {row['is_success']}. Errors: {row['metric_errors']}.\n"
            context_str += progress_summary
        
        if not context_str:
            return []
            
        return [Document(page_content=context_str)]
