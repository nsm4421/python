import sqlite3
# conn = sqlite.connect(':momory:')

# Connet to database
conn = sqlite3.connect('customer.db')

# create cursur
c = conn.cursor()

# Create Table
# datatype : NULL, Integer, Real, Text, Blob
c.execute("""
          CREATE TABLE customers (name text,
                                  email text,
                                  number text               
              )
          """)

# Commit
conn.commit()

# Close connection
conn.close()