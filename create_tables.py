import config
from modules import database as db
import os

def create_tables():

    # remove database file if already existt
    if os.path.exists(config.db_name):
        os.remove(config.db_name)

    admin_ = db.Admin(config.db_name)

    print("Creating admin table...")
    admin_.create_table()
    admin_.create_user_password(config.admin_username, config.admin_password)

    print("Creating user table...")
    user_ = db.User(config.db_name)
    user_.create_table()

    print("Creating object_detection table...")
    bugs_ = db.BeesDB(config.db_name)
    bugs_.create_table()

    # remove existing image upload_files
    for filename in os.listdir(config.bees_upload_dir):
        os.remove(os.path.join(config.bees_upload_dir, filename))

    for filename in os.listdir(config.bees_pred_dir):
        os.remove(os.path.join(config.bees_pred_dir, filename))

    for filename in os.listdir(config.insect_upload_dir):
        os.remove(os.path.join(config.insect_upload_dir, filename))

    for filename in os.listdir(config.insect_pred_dir):
        os.remove(os.path.join(config.insect_pred_dir, filename))

create_tables()
