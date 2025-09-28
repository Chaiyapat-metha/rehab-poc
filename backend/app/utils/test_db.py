import psycopg2
conn = psycopg2.connect(
    dbname="rehab_db",
    user="nonny",
    password="nonny",
    host="localhost",
    port=5433
)
cur = conn.cursor()
cur.execute("SELECT NOW();")
print(cur.fetchone())
conn.close()
