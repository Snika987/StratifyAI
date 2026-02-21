from Database.session import get_connection
import os    

try:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT 1;")
    result = cur.fetchone()
    print("DB Connected Successfully:", result)

    cur.close()
    conn.close()

except Exception as e:
    print("Connection Failed:", e)


