import sqlite3

# Connet to database
conn = sqlite3.connect('customer.db')

# create cursur
c = conn.cursor()

customers_info = [('구겸', '구겸@naver.com', '1'),
                 ('봉식', '봉식@naver.com', '2'),
                 ('재준', '재준@naver.com', '3'),
                 ('덕이', '덕이@naver.com', '4')]


c.executemany("INSERT INTO customers VALUES (?,?,?)", customers_info)

# Commit
conn.commit()

# Close connection
conn.close()