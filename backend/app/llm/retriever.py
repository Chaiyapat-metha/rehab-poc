from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List

from app.utils import db

class RehabDbRetriever(BaseRetriever):
    """A custom retriever that fetches a user's progress from our database."""
    user_id: str

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        The core logic of the retriever.
        It runs SQL queries based on the user's question.
        """
        # A simple keyword-based logic to decide which data to fetch.
        # A more advanced system would use embeddings or another LLM to parse the query.
        
        context_str = ""
        
        # --- Logic to fetch exercise list ---
        if "ท่า" in query or "exercise" in query:
            exercises = db.get_existing_exercise_names()
            if exercises:
                context_str += f"Available exercises: {', '.join(exercises)}.\n"

        # --- Logic to fetch user progress ---
        # For this PoC, we fetch the 5 most recent labeled frames.
        sql = """
        SELECT s.exercise_name, f.labels, f.time
        FROM frames f
        JOIN sessions s ON f.session_id = s.session_id
        WHERE s.user_id = %s AND f.labels IS NOT NULL
        ORDER BY f.time DESC
        LIMIT 5;
        """
        recent_progress = db.execute_query(sql, (self.user_id,))
        if recent_progress:
            progress_summary = "\nRecent User Performance:\n"
            for row in recent_progress:
                exercise_name, labels, time = row
                # Convert label dict to a readable string
                label_str = f"- At {time.strftime('%Y-%m-%d %H:%M')}, during '{exercise_name}', the pose was classified as '{labels.get('class')}' with severity {labels.get('severity', {})}.\n"
                progress_summary += label_str
            context_str += progress_summary
        
        if not context_str:
            return []
            
        # We return the entire context as a single Document
        return [Document(page_content=context_str)]
