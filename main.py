from modules import database
from config import db_name
from bees import bees_prediction
from insect import insect_prediction

db_obj = database.BeesDB(db_name)
vals = db_obj.get_values_by_date("Insects", "2022-02-25", "2022-02-28", 5, 1)
print(vals)
#admin_obj = database.Admin(db_name)
#user_obj = database.User(db_name)
#user_obj.create_table()
#admin_obj.create_table()
#admin_obj.create_user_password("admin", "admin123")
#res = admin_obj.is_admin("admin", "admin123")
#print(res)
#db_obj.create_table()

#tmp_record = {"source_img_name": "a.jpg", "pred_img_name": "b.jpg", "insect_counts": "5", "category": "Bees"}
#db_obj.insert_values("object_detection", **tmp_record)

#flag, rows = db_obj.get_values_by_category("object_detection", "Bees")
#num_pages = db_obj.get_total_pages("object_detection", "Bees", 1)
#print(rows)
#print(num_pages)

#filename, count = bees_prediction(filename="IMG-20211217-WA0027.jpg", src_img_path=UploadImgDir, dst_img_path=PredictedImgDir)
#print(count)


#img_name = "IMG_0026.jpg"
#src_img_path = "C:/Users/ankur/Downloads/NewImages/arjun_insects/arjun_insects/images"
#prediction_path = "C:/Users/ankur/Downloads/NewImages/arjun_insects/arjun_insects/images/pred"

#filename, insect_count, egg_count = insect_prediction(img_name, src_img_path, prediction_path)
#print(filename)
#print(insect_count)
#print(egg_count)
