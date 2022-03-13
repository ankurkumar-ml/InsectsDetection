from flask import Flask, request, render_template
from flask import session
from flask import redirect, url_for
import os, time
from config import *
from bees import bees_prediction
from insect import insect_prediction
from modules import database
from modules.imgs import get_images, get_images_by_date
from flask_session import Session

# create directories
if not os.path.exists(bees_upload_dir):
    os.makedirs(bees_upload_dir)

if not os.path.exists(bees_pred_dir):
    os.makedirs(bees_pred_dir)

if not os.path.exists(insect_upload_dir):
    os.makedirs(insect_upload_dir)

if not os.path.exists(insect_pred_dir):
    os.makedirs(insect_pred_dir)

# create an object of BeesDB
db_obj = database.BeesDB(db_name)
admin_obj = database.Admin(db_name)
user_obj = database.User(db_name)


app = Flask(__name__)
app.secret_key = "goodbugscount"


@app.route("/upload", methods=['GET', 'POST'])
def upload():

    if request.method == 'POST':
        upload_files = request.files.getlist('file[]')
        category = request.form.get('category')

        img_exists_count = 0
        already_files = os.listdir(bees_upload_dir)

        for file in upload_files:

            if file.filename in already_files:
                img_exists_count += 1
            else:
                file.save(os.path.join(bees_upload_dir, file.filename))
                filename, count = bees_prediction(filename=file.filename, src_img_path=bees_upload_dir, dst_img_path=bees_pred_dir)
                record = {"source_img_name": filename, "pred_img_name": filename, "insect_counts": count,
                              "category": category}
                db_obj.insert_values("object_detection", **record)

        msg = None

        if img_exists_count != 0:
            msg = f"{img_exists_count} images already exists!"

        return redirect(url_for("index", msg=msg))

    return render_template("upload.html")

@app.route("/admin_update_users/<int:id>", methods=["GET", "POST"])
def admin_update_users(id):

    if not session.get('admin_username'):
        return redirect("/admin_login")

    update_form_msg = None

    # get old username and password by id
    res = user_obj.get_user_by_id(id)
    old_username = res[0][0]
    old_password = res[0][1]

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        confirmpassword = request.form.get("confirmpassword")

        if password != confirmpassword:
            update_form_msg = "Password and confirm password does not match."
            return render_template("update_user.html", old_username=old_username, old_password=old_password, update_form_msg=update_form_msg)

        is_updated =  user_obj.update_user_password(id, username, password)

        if is_updated:
            update_form_msg = "Username and password updated successfully."
            session["update_success_msg"] = update_form_msg
            return redirect(url_for("admin_manage_users"))
        else:
            update_form_msg = "Username and password not updated."

    return render_template("update_user.html", old_username=old_username, old_password=old_password,
                           update_form_msg=update_form_msg, id=id)


@app.route("/admin_manage_users", defaults={'page': 1}, methods=['GET', 'POST'])
@app.route("/admin_manage_users/<int:page>", methods=['GET', 'POST'])
def admin_manage_users(page):

    if not session.get('admin_username'):
        return redirect("/admin_login")

    user_form_msg = None
    user_table_msg = None
    prev_page = page - 1
    next_page = page + 1
    users_per_page = 7
    total_pages = user_obj.get_total_pages(users_per_page)
    users = user_obj.get_users(users_per_page, page)

    ## Start delete user code section
    if "del_username" in request.args.keys() and "del_password" in request.args.keys():
        del_username = request.args.get("del_username")
        del_password = request.args.get("del_password")

        is_deleted = user_obj.delete_user(del_username, del_password)

        if is_deleted:
            user_table_msg = "User deleted successfully."
            users = user_obj.get_users(users_per_page, page)
        else:
            user_table_msg = "User not deleted."

    ## End delete user code section

    if request.method == "POST" and request.form['submit'] == "Submit":

        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            user_form_msg = "Password and Confirm password are not the same."
        elif user_obj.is_user(username):
            user_form_msg = "Username already exists."
        else:
            user_obj.create_user_password(username, password)
            user_form_msg = "User created successfully."

        users = user_obj.get_users(users_per_page, page)

        return render_template("admin_users.html", prev_page=prev_page, next_page=next_page,
                               total_pages=total_pages, user_form_msg=user_form_msg, users=users,
                               user_table_msg=user_table_msg)

    return render_template("admin_users.html", prev_page=prev_page, next_page=next_page,
                           total_pages=total_pages, user_form_msg=user_form_msg, users=users,
                           user_table_msg=user_table_msg)


@app.route("/admin_panel", methods=['GET', 'POST'], defaults={'page': 1})
@app.route("/admin_panel/<int:page>", methods=['GET', 'POST'])
def admin_panel(page):

    upload_form_msg = None
    upload_batch_size = 10
    category_opt = "Bees"

    if "category_opt" in request.args.keys():
        category_opt = request.args.get("category_opt")


    if not session.get('admin_username'):
        return redirect("/admin_login")

    items_per_page = 1
    total_pages = db_obj.get_total_pages("object_detection", category_opt, items_per_page)

    prev_page = page - 1
    next_page = page + 1
    msg, img_rows = get_images(category_opt, items_per_page, page)

    if request.method == "POST" and "upload" in request.form.keys():
        upload_files = request.files.getlist('file[]')
        category = request.form.get('category')

        if len(upload_files) > upload_batch_size:
            upload_form_msg = "More then 10 images are not allowed."
        else:
            uploaded_files = None
            if category == "Bees":
                uploaded_files = os.listdir(bees_upload_dir)
            else:
                uploaded_files = os.listdir(insect_upload_dir)

            exists_count = 0  # Count - how many images already have been uploaded
            for file in upload_files:

                if file.filename in uploaded_files:
                    exists_count += 1
                else:
                    if category == "Bees":
                        file.save(os.path.join(bees_upload_dir, file.filename))
                        filename, count = bees_prediction(filename=file.filename, src_img_path=bees_upload_dir, dst_img_path=bees_pred_dir)
                        record = {"source_img_name": filename, "pred_img_name": filename, "insect_counts": count,
                                      "category": category}
                        db_obj.insert_values("object_detection", **record)
                    else:
                        file.save(os.path.join(insect_upload_dir, file.filename))
                        filename, count = insect_prediction(filename=file.filename, src_img_path=insect_upload_dir,
                                                          dst_img_path=insect_pred_dir)
                        record = {"source_img_name": filename, "pred_img_name": filename, "insect_counts": str(count),
                                  "category": category}
                        db_obj.insert_values("object_detection", **record)

            if exists_count != 0:
                upload_form_msg = f"{exists_count} images already exists."
            else:
                upload_form_msg = "All images uploaded successfully."

        return render_template("index.html", img_rows=img_rows, prev_page=prev_page, next_page=next_page,
                               total_pages=total_pages,upload_form_msg=upload_form_msg,
                               category_name=category_opt.lower(), category_opt=category_opt)

    return render_template("index.html", img_rows=img_rows, prev_page=prev_page, next_page=next_page,
                           total_pages=total_pages, upload_form_msg=upload_form_msg,
                           category_name=category_opt.lower(), category_opt=category_opt)

@app.route("/", methods=['GET', 'POST'])
@app.route("/login", methods=['GET', 'POST'])
def login():

    login_msg = None

    if request.method == 'POST':
        username = request.form.get("username")
        password = request.form.get("password")

        is_user = user_obj.is_user(username, password)

        if is_user:
            session['user_username'] = username
            session['user_password'] = password

            # redirect to user dashboard
            return redirect(url_for("user_dashboard"))
        else:
            login_msg = "Incorrect username or password."

    return render_template("login.html", login_msg=login_msg)

@app.route("/admin_login", methods=['GET', 'POST'])
def admin_login():

    err_msg = None

    if request.method == 'POST':
        username = request.form.get("username")
        password = request.form.get("password")

        is_admin = admin_obj.is_admin(username, password)
        print(is_admin, username, password)
        if is_admin:
            session['admin_username'] = username
            session['admin_password'] = password
            # redirect to admin-panel
            return redirect(url_for("admin_panel"))
        else:
            err_msg = "Incorrect username or password."

    return render_template("admin_login.html", err_msg=err_msg)

@app.route("/admin_logout")
def admin_logout():
    session['admin_username'] = None
    session['admin_password'] = None
    session["update_success_msg"] = None
    return redirect("/admin_login")

@app.route("/user_logout")
def user_logout():
    session['user_username'] = None
    session['user_password'] = None
    return redirect("/login")

@app.route("/user_dashboard", defaults={'page': 1}, methods=['GET', 'POST'])
@app.route("/user_dashboard/<int:page>", methods=['GET', 'POST'])
def user_dashboard(page):

    if not session.get('user_username'):
        return redirect("/login")

    upload_form_msg = None
    upload_batch_size = 10
    category_opt = "Bees"

    if "category_opt" in request.args.keys():
        category_opt = request.args.get("category_opt")

    items_per_page = 3
    total_pages = db_obj.get_total_pages("object_detection", category_opt, items_per_page)

    prev_page = page - 1
    next_page = page + 1
    msg, img_rows = get_images(category_opt, items_per_page, page)

    if request.method == "POST" and "search" in request.form.keys():
        start_date = request.form.get("start_date")
        end_date = request.form.get("end_date")
        total_pages = db_obj.get_total_pages_by_date(category_opt, items_per_page, start_date, end_date)
        img_rows = get_images_by_date(category_opt, start_date, end_date, items_per_page, page)

    if request.method == "POST" and "upload" in request.form.keys():
        upload_files = request.files.getlist('file[]')
        category = request.form.get('category')

        if len(upload_files) > upload_batch_size:
            upload_form_msg = "More then 10 images are not allowed."
        else:
            uploaded_files = None
            if category == "Bees":
                uploaded_files = os.listdir(bees_upload_dir)
            else:
                uploaded_files = os.listdir(insect_upload_dir)

            exists_count = 0  # Count - how many images already have been uploaded
            for file in upload_files:

                if file.filename in uploaded_files:
                    exists_count += 1
                else:
                    if category == "Bees":
                        file.save(os.path.join(bees_upload_dir, file.filename))
                        filename, count = bees_prediction(filename=file.filename, src_img_path=bees_upload_dir, dst_img_path=bees_pred_dir)
                        record = {"source_img_name": filename, "pred_img_name": filename, "insect_counts": count,
                                      "category": category}
                        db_obj.insert_values("object_detection", **record)
                    else:
                        file.save(os.path.join(insect_upload_dir, file.filename))
                        filename, count = insect_prediction(filename=file.filename, src_img_path=insect_upload_dir,
                                                          dst_img_path=insect_pred_dir)
                        record = {"source_img_name": filename, "pred_img_name": filename, "insect_counts": str(count),
                                  "category": category}
                        db_obj.insert_values("object_detection", **record)

            if exists_count != 0:
                upload_form_msg = f"{exists_count} images already exists."
            else:
                upload_form_msg = "Images uploaded successfully."

        return render_template("user_dashboard.html", img_rows=img_rows, prev_page=prev_page, next_page=next_page,
                               total_pages=total_pages,upload_form_msg=upload_form_msg,
                               category_name=category_opt.lower(), category_opt=category_opt)

    #print(img_rows, page)
    return render_template("user_dashboard.html", img_rows=img_rows, prev_page=prev_page, next_page=next_page,
                    total_pages=total_pages, upload_form_msg=upload_form_msg,
                    category_name=category_opt.lower(), category_opt=category_opt)
