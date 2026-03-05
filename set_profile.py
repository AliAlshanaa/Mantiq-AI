import sqlite3
import os

# Database path
DB_PATH = "data/mantiq_enterprise.db"

def initialize_database_if_missing():
    """Ensures the database and tables exist before updating preferences."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create User Profile Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_profile (
            user_id INTEGER PRIMARY KEY,
            preferred_tone TEXT DEFAULT 'Professional',
            formatting_style TEXT DEFAULT 'Detailed Markdown',
            language TEXT DEFAULT 'Arabic/English',
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Insert default user if not exists
    cursor.execute("INSERT OR IGNORE INTO user_profile (user_id) VALUES (1)")
    
    # Create Task History Table
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

def update_preferences():
    print("\n--- ⚙️ Mantiq-AI: User Profile Manager ---")
    
    # Step 0: Ensure DB is ready
    initialize_database_if_missing()

    # 1. Select Tone
    print("Select Preferred Tone:")
    print("1. Professional & Formal (Default)")
    print("2. Academic & Data-Driven")
    print("3. Creative & Engaging")
    print("4. Concise & Direct")
    tone_choice = input("Choice (1-4) or type your own: ").strip()

    tone_map = {
        "1": "Professional and Formal",
        "2": "Academic, focused on data and citations",
        "3": "Creative, storytelling approach",
        "4": "Very concise and direct (bullet points only)"
    }
    final_tone = tone_map.get(tone_choice, tone_choice)

    # 2. Select Formatting
    print("\nSelect Formatting Style:")
    print("1. Detailed Markdown with Headers (Default)")
    print("2. Executive Summary Style (One-pager)")
    print("3. Technical Documentation (Code blocks & Tables)")
    format_choice = input("Choice (1-3) or type your own: ").strip()

    format_map = {
        "1": "Detailed Markdown with hierarchical headers and bullet points",
        "2": "Executive Summary format: Brief, high-impact paragraphs",
        "3": "Technical whitepaper style with tables and structured data blocks"
    }
    final_format = format_map.get(format_choice, format_choice)

    # 3. Update Database
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE user_profile 
            SET preferred_tone = ?, formatting_style = ?, last_updated = CURRENT_TIMESTAMP
            WHERE user_id = 1
        """, (final_tone, final_format))
        conn.commit()
        conn.close()
        
        print("\n" + "="*40)
        print("✅ SUCCESS: User Profile Initialized & Updated!")
        print(f"👉 Tone: {final_tone}")
        print(f"👉 Style: {final_format}")
        print("="*40)

    except Exception as e:
        print(f"--- ❌ ERROR: {e} ---")

if __name__ == "__main__":
    update_preferences()