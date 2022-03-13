import config
from modules.database import BeesDB
import os

def get_images(category, items_per_page, page_num):
    db_obj = BeesDB(config.db_name)

    msg = None
    rows = db_obj.get_values_by_category("object_detection", category, items_per_page, page_num)

    if len(rows) == 0:
        msg = "Please upload some photos to view predictions."
        return msg, rows

    new_rows = []

    for row in rows:
        id = row[0]
        src_img_name = row[1]
        pred_img_name = row[2]
        bees_count = row[3]
        date_val =  row[-1]

        new_row = {'src_img_name': src_img_name,
                   'pred_img_name': pred_img_name,
                   'bees_count': bees_count,
                   'uploaded_date': date_val}
        new_rows.append(new_row)

    new_rows.reverse()

    return msg, new_rows


def get_images_by_date(category, start_date, end_date, items_per_page, page_num):
    db_obj = BeesDB(db_name)

    rows = db_obj.get_values_by_date(category, start_date, end_date, items_per_page, page_num)

    new_rows = []

    for row in rows:
        id = row[0]
        src_img_name = row[1]
        pred_img_name = row[2]
        bees_count = row[3]
        date_val = row[-1]

        new_row = {'src_img_name': src_img_name,
                   'pred_img_name': pred_img_name,
                   'bees_count': bees_count,
                   'uploaded_date': date_val}
        new_rows.append(new_row)

    new_rows.reverse()

    return new_rows
