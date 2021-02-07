import sqlite3

# Connet to database
conn = sqlite3.connect('customer.db')

# create cursur
c = conn.cursor()

# Select Query
c.execute("""UPDATE customers SET name = '구갬'
          WHERE name = '승연'
          """)
          
fetch_all = c.fetchall()
for item in fetch_all:
    print(item)
    
# Commit
conn.commit()

# Close connection
conn.close()