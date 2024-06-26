from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import pandas as pd
from flask_mysqldb import MySQL
from flask_paginate import Pagination
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from flask_bcrypt import Bcrypt
import logging

app = Flask(__name__)
app.secret_key= 'your-secret-key'

# Initialize Bcrypt
bcrypt = Bcrypt(app)

#Meghubungkan Mysql
app.config['MYSQL_HOST'] ='localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'prediksi_churn'

mysql = MySQL(app)

#mapping
mappingServiceTypes = {
    'Corporate':1,
    'Retail':0
}

mappingPacketService = {
    'Permana Home':0, 
    'Permana Link':1, 
    'Permana Dedicated':2, 
    'Permana Hosta':3,
    'Permana Colocation':4, 
    'Permana Metro':5
}

mappingMediaTransmisi = {
    'Wireless':0, 
    'Fiber Optic':1, 
    'Ethernet':2, 
    'Direct Kabel':3
}

mappingState= {
    'Kepulauan Riau':0, 
    'DKI Jakarta':1, 
    'Aceh':2, 
    'Jawa Barat':3, 
    'Jawa Tengah':4,
    'Jakarta':5, 
    'Sumatera Utara':6, 
    'Riau':7, 
    'Banten':8, 
    'Jawa Timur':9,
    'Kalimantan Timur':10, 
    'Kalimantan Barat':11, 
    'Sulawesi Selatan':12
}

mappingPartner = {'Yes':1, 'No':0}

mappingContract = {
    'Yearly':1, 
    'Monthly':0
}

mappingComplaint = {'Yes':1, 'No':0}

mappingChurn = {'Yes':1, 'No':0}

#convert
def extract_number(string):
    try:
        return int(string.split()[0])
    except (ValueError, AttributeError):
        return 0
    
def convertServiceTypes(st):
    role = {
        0: "Corporate",
        1: "Retail"
    }
    return role.get(st, "unknown")

def convertPacketService(ps):
    role = {
        0:"Permana Home", 
        1:'Permana Link', 
        2:'Permana Dedicated', 
        3:'Permana Hosta',
        4:'Permana Colocation', 
        5:'Permana Metro'
    }
    return role.get(ps, "unknown")

def convertMediaTransmisi(mt):
    role = {
        0:'Wireless', 
        1:'Fiber Optic', 
        2:'Ethernet', 
        3:'Direct Kabel'
    }
    return role.get(mt, "unknown")

def convertState(stt):
    role = {
        0:'Kepulauan Riau', 
        1:'DKI Jakarta', 
        2:'Aceh', 
        3:'Jawa Barat', 
        4:'Jawa Tengah',
        5:'Jakarta', 
        6:'Sumatera Utara', 
        7:'Riau', 
        8:'Banten', 
        9:'Jawa Timur',
        10:'Kalimantan Timur', 
        11:'Kalimantan Barat', 
        12:'Sulawesi Selatan'
    }
    return role.get(stt, "unknown")

def convertPartner(pn):
    role = {
        0: "No",
        1: "Yes"
    }
    return role.get(pn, "unknown")

def convertContract(ct):
    role = {
        0: "No",
        1: "Yes"
    }
    return role.get(ct, "unknown")

def convertComplaint(cp):
    role = {
        0: "No",
        1: "Yes"
    }
    return role.get(cp, "unknown")

def convertChurn(ch):
    role = {
        0: "No",
        1: "Yes"
    }
    return role.get(ch, "unknown")

app.jinja_env.globals.update(convertst=convertServiceTypes,convertps=convertPacketService,convertMT=convertMediaTransmisi,convertstt=convertState,convertpt=convertPartner,convertcp=convertComplaint,convertch=convertChurn)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        try:
            cur = mysql.connection.cursor()
            cur.execute("SELECT username, nama, password FROM tbl_user WHERE username = %s", (username,))
            user = cur.fetchone()
            cur.close()

            if user and bcrypt.check_password_hash(user[2], password):
                session['nama'] = user[1]  # Set session jika login sukses
                return redirect(url_for('dashboard'))  # Arahkan ke dashboard setelah login sukses
            else:
                flash('ERROR: Username dan Password Anda Salah')
                return render_template('login.html')

        except Exception as e:
            logging.error(f"An error occurred during login: {e}")
            flash(f'ERROR: An error occurred ({e})')
            return render_template('login.html')

    return render_template('login.html')


def create_hashed_password(password):
    return bcrypt.generate_password_hash(password).decode('utf-8')

@app.route('/insert', methods=['POST'])
def insert():
    if request.method == "POST":
        flash("Register Berhasil")
        nama = request.form['nama']
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']
        level = request.form['level']
    
        hashed_password = create_hashed_password(password)
        
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO tbl_user (nama, email, username, password, level) VALUES (%s, %s, %s, %s, %s)", (nama, email, username, hashed_password, level))
        mysql.connection.commit()
        cur.close()
        return redirect(url_for('user'))
    
@app.route('/delete/<string:id_data>', methods = ['POST','DELETE'])
def delateuser(id_data):
    if (request.form['_method'] == 'DELETE'):
        flash("Delete Data Berhasil")
        cur = mysql.connection.cursor()
        cur.execute("DELETE FROM tbl_user WHERE id = %s", (id_data,))
        mysql.connection.commit()
        cur.close()
        return redirect (url_for('user'))

@app.route('/user/reset/<user_id>', methods=['POST', 'PUT'])
def putUserReset(user_id):
    password = request.form['password']
    password2 = request.form['password2']
    
    if password == password2:
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        flash("Change Password Berhasil")
        cur = mysql.connection.cursor()
        cur.execute("""UPDATE tbl_user SET password = %s WHERE id = %s""", (hashed_password, user_id))
        mysql.connection.commit()
        cur.close()
        return redirect(url_for('user'))
    else:
        flash("Password tidak cocok")
        return redirect(url_for('user'))

#####
@app.route('/forgot_pass')
def forgot_pass():
    return render_template('forgot-password.html')

@app.route('/dashboard')
def dashboard():
    if 'nama' in session:
        try:
            cur = mysql.connection.cursor()
            cur.execute('SELECT COUNT(*) FROM tbl_testing')
            uji = cur.fetchone()[0]  # Mengambil nilai COUNT(*)
            cur.execute('SELECT COUNT(*) FROM tbl_training')
            latih = cur.fetchone()[0]  # Mengambil nilai COUNT(*)
            cur.close()
            
            return render_template('dashboard.html', nama=session['nama'], testing=uji, training=latih)
        
        except Exception as e:
            return render_template('dashboard.html', nama=session['nama'])
    return redirect(url_for('login'))

@app.route ('/datatraining', methods = ['GET'])
def datatraining ():    
    if 'nama' in session:
        cur = mysql.connection.cursor()
        cur.execute ("SELECT * From tbl_training")
        data = cur.fetchall()
        cur.close()

        return render_template('training-prediksi.html', nama=session['nama'], tbl_training = data)
    else:
        return render_template('login.html')
        
@app.route ('/datatraining', methods = ['POST'])
def inputdatatraining ():
    if request.method == 'POST':
        try:
            data = request.files['file']
            data = pd.read_csv(data)

            # Mengganti nilai string kosong ('') dengan NaN
            data.replace('', pd.NA, inplace=True)
            data.replace('-', pd.NA, inplace=True)

            # Menghapus baris dengan nilai kosong atau NaN di salah satu kolom
            data = data.dropna(how='any')

            data['Nilai Service_types'] = data['Service_types'].replace(mappingServiceTypes)
            data['Nilai Packet_service'] = data['Packet_service'].replace(mappingPacketService)
            data['Nilai Media_transmisi'] = data['Media_transmisi'].replace(mappingMediaTransmisi)
            data['Nilai State'] = data['State'].replace(mappingState)
            data['Nilai Partner'] = data['Partner'].replace(mappingPartner)
            data['Nilai Contract'] = data['Contract'].replace(mappingContract)
            data['Nilai Complaint'] = data['Complaint'].replace(mappingComplaint)
            data['Nilai Churn'] = data['Complaint'].replace(mappingChurn)

            cur = mysql.connection.cursor()
            # Iterasi melalui setiap baris DataFrame dan menyimpan data ke database
            for index, row in data.iterrows():
                # Menjalankan kueri untuk menyimpan data
                cur.execute("INSERT INTO tbl_training (nama, service_type, packet_service, media_transmisi, bandwidth, state, partner, contract, complaint, churn) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)", 
                            (
                                row['Nama Pelanggan'], 
                                row['Nilai Service_types'], 
                                row['Nilai Packet_service'], 
                                row['Nilai Media_tranmisi'], 
                                row['Nilai State'],
                                row['Nilai Partner'],
                                row['Nilai Contract'],
                                row['Nilai Complaint'],
                                row['Nilai Churn'],
                                ))
            mysql.connection.commit()
            cur.close()
            flash("Data Training Berhasil DiUpload")
            return redirect(url_for('datatraining'))
        except Exception as e:
            flash("ERROR: Terjadi Kesalahan Saat Menambah Data Training")
            return redirect(url_for('datatraining'))
   
@app.route('/deletedatatraining/<string:id_data>', methods = ['POST','DELETE'])
def deletedatatraining(id_data):
    if (request.form['_method'] == 'DELETE'):
        flash("Delete Data Training Berhasil")
        cur = mysql.connection.cursor()
        cur.execute("DELETE FROM tbl_training WHERE id_training = %s", (id_data,))
        mysql.connection.commit()
        cur.close()
        return redirect (url_for('datatraining'))    

@app.route('/prediksi')
def prediksi():
    return render_template('input-prediksi.html')
    

@app.route('/history')
def history(): 
    if 'nama' in session:
        cur = mysql.connection.cursor()
        cur.execute ("SELECT * From tbl_prediksi")
        data =cur.fetchall()
        cur.close()
        return render_template('history-prediksi.html', nama=session['nama'], tbl_datatesting = data)
    else:
        return render_template('login.html')

@app.route('/deletedataprediksi/<string:id_data>', methods = ['POST','DELETE'])
def delatedataprediksi(id_data):
    if (request.form['_method'] == 'DELETE'):
        flash("Delete Data Testing Berhasil")
        cur = mysql.connection.cursor()
        cur.execute("DELETE FROM tbl_prediksi WHERE id_prediksi = %s", (id_data,))
        mysql.connection.commit()
        cur.close()
        return redirect (url_for('history'))

@app.route ('/user')
def user ():
    if 'nama' in session:
        cur = mysql.connection.cursor()
        cur.execute ("SELECT * From tbl_user")
        data =cur.fetchall()
        cur.close()
        return render_template('user.html', tbl_users = data, nama=session['nama'] )
    else:
        return render_template('login.html')

@app.route('/report')
def report():
    return render_template('report.html')

@app.route('/logout')
def logout():
    if 'nama' in session:
        session.pop('nama', None)
    return redirect('/login')

@app.route('/tes')
def tes():
    return render_template('tes.html')


if __name__ == '__main__':
    app.run(debug=True)

