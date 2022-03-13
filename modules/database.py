import math
from datetime import datetime, timedelta
import sqlite3

class Admin:
    def __init__(self, db_name):
        self.db_name = db_name

    def connect(self):
        conn = sqlite3.connect(self.db_name)
        return conn

    def create_table(self):
        # connect to the database
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute('''
                  CREATE TABLE admin
                  (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT NOT NULL,
                  password TEXT NOT NULL,
                  created_at TEXT DEFAULT CURRENT_TIMESTAMP)''')

        conn.commit()
        print("Table created successfully.")

        conn.close()

    def create_user_password(self, username, password):
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute("insert into admin (username, password) values (?, ?)", (username, password, ))
        conn.commit()

        print(f"Created\nusername: {username}\npassword: {password}")

        conn.close()

    def is_admin(self, username, password):
        conn = self.connect()
        cursor = conn.cursor()

        res = cursor.execute("select * from admin where username = ? and password = ?", (username, password, ))

        rows = res.fetchall()

        conn.close()

        if len(rows) != 0:
            return True

        return False


class User:
    def __init__(self, db_name):
        self.db_name = db_name


    def connect(self):
        conn = sqlite3.connect(self.db_name)
        return conn

    def create_table(self):
        conn = self.connect()

        cursor = conn.cursor()
        cursor.execute('''
                          CREATE TABLE user
                          (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          username varchar(100) NOT NULL,
                          password varchar(100) NOT NULL,
                          created_at TEXT DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()
        print("Table created successfully.")
        conn.close()

    def create_user_password(self, username, password):
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute("insert into user (username, password) values (?, ?)", (username, password, ))
        conn.commit()

        print(f"Created\nusername: {username}\npassword: {password}")

        conn.close()

    def get_user_by_id(self, id):
        conn = self.connect()
        cursor = conn.cursor()

        res = cursor.execute(f"select username, password from user where id = ?", (id, ))

        rows = res.fetchall()

        conn.close()

        return rows

    def update_user_password(self, id, Newusername, Newpassword):
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute("UPDATE user SET username = ?, password = ? where id = ?", (Newusername, Newpassword, id))
        conn.commit()

        if cursor.rowcount:
            return True

        return False

    def delete_user(self, username, password):
        delete_flag = False

        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute("delete from user where username = ? and password = ?", (username, password, ))
        conn.commit()

        if cursor.rowcount:
            delete_flag = True
            return delete_flag

        conn.close()

        return delete_flag

    def is_user(self, username, password=None):
        conn = self.connect()
        cursor = conn.cursor()

        if password == None:
            res = cursor.execute("select * from  user where username = ?", (username, ))
        else:
            res = cursor.execute("select * from  user where username = ? and password = ?", (username, password, ))

        rows = res.fetchall()

        conn.close()

        if len(rows) != 0:
            return True

        return False

    def get_users(self, items_per_page, page_num):
        conn = self.connect()
        cursor = conn.cursor()

        offset = (page_num - 1) * items_per_page

        res = cursor.execute(f"select * from user limit {items_per_page} offset {offset}")

        rows = res.fetchall()

        conn.close()

        return rows

    def get_total_pages(self, items_per_page):
        conn = self.connect()
        cursor = conn.cursor()

        res = cursor.execute(f"select count(*) from user")

        rows = res.fetchall()

        conn.close()

        total_items = rows[0][0]

        if total_items > 0 and total_items > items_per_page:
            num_pages = math.ceil(total_items / items_per_page)
            return num_pages

        return 1


class BeesDB:
    def __init__(self, db_name):
        self.db_name = db_name

    def create_table(self):
        # connect to the database
        conn = self.connect()

        cursor = conn.cursor()
        cursor.execute('''
          CREATE TABLE object_detection
          (id INTEGER PRIMARY KEY AUTOINCREMENT,
          source_img_name TEXT NOT NULL,
          pred_img_name TEXT NOT NULL,
          insect_counts TEXT NOT NULL,
          category TEXT NOT NULL,
          uploaded_by TEXT,
          created_at TEXT DEFAULT CURRENT_TIMESTAMP)''')

        conn.commit()
        print("Table created successfully.")

        self.disconnect(conn)

    def connect(self):
        conn = sqlite3.connect(self.db_name)
        return conn

    def disconnect(self, conn):
        conn.close()

    def insert_values(self, table_name, **kwargs):

        is_exists = False

        conn = self.connect()
        cursor = conn.cursor()

        res = cursor.execute(f"select * from {table_name} where source_img_name = ?", (kwargs['source_img_name'],))

        if res.fetchall():
            is_exists = True
            return is_exists

        cursor.execute(f"insert into {table_name} (source_img_name, pred_img_name, insect_counts, category) \
                    values (?, ?, ?, ?)", (kwargs['source_img_name'], kwargs['pred_img_name'],
                                        kwargs['insect_counts'], kwargs['category'], ))

        conn.commit()

        print("data inserted successfully.")

        self.disconnect(conn)

        return is_exists

    def get_values_by_date(self, category, start_date, end_date, items_per_page, page_num):
        conn = self.connect()
        cursor = conn.cursor()

        offset = (page_num - 1) * items_per_page
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        end_date += timedelta(days=1)
        end_date = end_date.strftime("%Y-%m-%d")

        res = cursor.execute(f"select * from object_detection where created_at >= ? and created_at < ? and category = ? limit {items_per_page} offset {offset}",
                          (start_date, end_date, category,))

        rows = res.fetchall()

        self.disconnect(conn)

        return rows

    def get_values_by_category(self, table_name, category, items_per_page, page_num):
        conn = self.connect()
        cursor = conn.cursor()

        offset = (page_num - 1) * items_per_page

        res = cursor.execute(f"select * from {table_name} where category = ? limit {items_per_page} offset {offset}", (category,))

        rows = res.fetchall()

        self.disconnect(conn)

        return rows

    def get_total_pages(self, table_name, category, items_per_page):
        conn = self.connect()
        cursor = conn.cursor()

        res = cursor.execute(f"select count(*) from {table_name} where category = ?", (category,))

        rows = res.fetchall()

        self.disconnect(conn)

        total_items = rows[0][0]

        if total_items > 0 and total_items > items_per_page:
            num_pages = math.ceil(total_items / items_per_page)
            return num_pages

        return 1

    def get_total_pages_by_date(self, category, items_per_page, start_date, end_date):
        conn = self.connect()

        cursor = conn.cursor()
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        end_date += timedelta(days=1)
        end_date = end_date.strftime("%Y-%m-%d")

        res  = cursor.execute(f"select count(*) from object_detection where created_at >= ? and created_at < ? and category = ?",
                          (start_date, end_date, category,))

        rows = res.fetchall()

        self.disconnect(conn)

        total_items = rows[0][0]

        if total_items > 0 and total_items > items_per_page:
            num_pages = math.ceil(total_items / items_per_page)
            return num_pages

        return 1
