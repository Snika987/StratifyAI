from Database.session import get_connection
import os

def run_migrations():
    conn = get_connection()
    cur = conn.cursor()
    
    with open("Database/schema_middleware.sql", "r") as f:
        sql = f.read()
    print("SQL BEING EXECUTED:\n", sql)
    
    try:
        cur.execute(sql)
        conn.commit()
        print("Migrations ran successfully.")
    except Exception as e:
        conn.rollback()
        print(f"Error running migrations: {e}")
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    run_migrations()
