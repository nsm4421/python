import sqlite3
# conn = sqlite.connect(':momory:')

# Connet to database
conn = sqlite3.connect('customer.db')

# create cursur
c = conn.cursor()

# Insert values
c.execute("INSERT INTO customers VALUES ('Karma','nsm4421@naver.com','1221')")

# Commit
conn.commit()

# Close connection
conn.close()