import sqlite3
import os

# Persistent database path within the project  directory
DB_PATH = "data/mantiq_enterprise.db"

class DatabaseManager:
    """
    Manages structured data persistence for Mantiq-AI, including user preferences 
    and task execution history. Utilizes SQLite for local, serverless storage.
    """
    def __init__(self):
        # Ensure the data directory exists before attempting to connect
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        self.init_db()

    def _get_connection(self):
        """
        Creates and returns a connection to the SQLite database.
        Sets row_factory to sqlite3.Row for dictionary-like access to columns.
        """
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  
        return conn

    def init_db(self):
        """
        Initializes the database schema if it doesn't already exist.
        Creates tables for User Profiles and historical task logs.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 1. User Profile Table: Stores long-term writing and formatting preferences
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profile (
                user_id INTEGER PRIMARY KEY,
                preferred_tone TEXT DEFAULT 'Professional',
                formatting_style TEXT DEFAULT 'Detailed Markdown',
                language TEXT DEFAULT 'Arabic/English',
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Insert default settings for the primary user if the table is empty
        cursor.execute("INSERT OR IGNORE INTO user_profile (user_id) VALUES (1)")
        
        # 2. Task History Table: Tracks agent activities for long-term memory analysis
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS task_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_description TEXT,
                model_used TEXT,
                completion_date DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        print("--- LOG: SQLite Database Initialized Successfully ---")

    # --- Profile Operations (User Preference Management) ---
    
    def get_user_profile(self) -> dict:
        """
        Retrieves the current user's preferences from the database.
        """
        conn = self._get_connection()
        profile = conn.execute("SELECT * FROM user_profile WHERE user_id = 1").fetchone()
        conn.close()
        return dict(profile) if profile else None

    def update_user_profile(self, tone: str = None, formatting: str = None):
        """
        Updates the user's preferred writing tone or formatting style.
        """
        conn = self._get_connection()
        if tone:
            conn.execute("""
                UPDATE user_profile 
                SET preferred_tone = ?, last_updated = CURRENT_TIMESTAMP
            """, (tone,))
        if formatting:
            conn.execute("""
                UPDATE user_profile 
                SET formatting_style = ?, last_updated = CURRENT_TIMESTAMP
            """, (formatting,))
            
        conn.commit()
        conn.close()
        print(f"--- LOG: Profile Updated (Tone: {tone}, Format: {formatting}) ---")

    # --- Task History Operations (Long-Term Memory Persistence) ---

    def log_task_completion(self, task_description: str, model_used: str):
        """
        NEW: Archives a completed task into the history table.
        This enables the AI to 'remember' what it has worked on in the past.
        """
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT INTO task_history (task_description, model_used)
                VALUES (?, ?)
            """, (task_description, model_used))
            conn.commit()
            print(f"--- LOG: Task archived successfully: {task_description[:30]}... ---")
        except Exception as e:
            print(f"--- ERROR: Failed to log task: {e} ---")
        finally:
            conn.close()

# Global singleton instance for easy access across agent nodes
db = DatabaseManager()