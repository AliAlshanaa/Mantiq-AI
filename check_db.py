import sqlite3
import os

def check_db():
    db_path = "data/mantiq_enterprise.db"
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # List tables
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cur.fetchall()
    print(f"Tables in {db_path}: {[t[0] for t in tables]}")
    
    # Check for some data
    for table in tables:
        t_name = table[0]
        cur.execute(f"SELECT COUNT(*) FROM {t_name}")
        count = cur.fetchone()[0]
        print(f"Table '{t_name}' has {count} rows.")
        
        if count > 0:
            cur.execute(f"SELECT * FROM {t_name} LIMIT 1")
            row = cur.fetchone()
            print(f"Sample row from '{t_name}': {row}")
            
    conn.close()

if __name__ == "__main__":
    check_db()
