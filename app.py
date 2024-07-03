from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_mysqldb import MySQL
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Bidirectional, LSTM, Dense
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import seaborn as sns
from flask_bcrypt import Bcrypt

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Konfigurasi MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'prediksi_churn02'

mysql = MySQL(app)

#melatih model
def train_model(data):
    try:
        # Preprocessing data
        categorical_columns = ['Service_types', 'Packet_service', 'Media_transmisi', 'State', 'Partner', 'Type_contract', 'Complaint', 'Churn']
        numerical_columns = ['Bandwidth']

        # Encode categorical data
        label_encoders = {}
        for col in categorical_columns:
            label_encoders[col] = LabelEncoder()
            data[col] = label_encoders[col].fit_transform(data[col])

        # Convert numerical data to float and normalize
        scaler = StandardScaler()
        data[numerical_columns] = scaler.fit_transform(data[numerical_columns].astype(float))

        X = data.drop('Churn', axis=1).values
        y = data['Churn'].values.ravel()  # Menggunakan ravel() pada y

        # Reshape X for LSTM
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Definisikan model BiLSTM
        model = Sequential()
        model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(X.shape[1], 1)))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Melatih model
        print("Starting model training...")
        model.fit(X, y, epochs=10, batch_size=32)
        print("Model training completed.")
        
        # Simpan model
        print("Saving model...")
        model.save('models/bilstm01.h5')
        print("Model saved successfully.")

        # Save the label encoders
        with open('models/label_encoders.pkl', 'wb') as f:
            pickle.dump(label_encoders, f)
    
        return 'Model trained and saved successfully.'

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

#fungsi predict model
def predict_model(input_data):
    try:
        print("Loading model...")
        model = load_model('models/bilstm01.h5')
        print("Model loaded successfully.")

        print("Loading label encoders...")
        with open('models/label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        print("Label encoders loaded successfully.")

        print("Preprocessing input data...")
        categorical_columns = ['Service_types', 'Packet_service', 'Media_transmisi', 'State', 'Partner', 'Type_contract', 'Complaint']
        numerical_columns = ['Bandwidth']

        # Encode categorical data
        for col in categorical_columns:
            if input_data[col] not in label_encoders[col].classes_:
                # Handle unseen labels
                input_data[col] = 'unknown'
                if 'unknown' not in label_encoders[col].classes_:
                    classes = np.append(label_encoders[col].classes_, 'unknown')
                    label_encoders[col].classes_ = classes
            input_data[col] = label_encoders[col].transform([input_data[col]])

        # Convert to DataFrame for easier handling
        input_df = pd.DataFrame([input_data])

        # Convert numerical data to float and normalize
        scaler = StandardScaler()
        input_df[numerical_columns] = scaler.fit_transform(input_df[numerical_columns].astype(float))

        # Ensure all data is float32
        input_df = input_df.astype('float32')

        # Reshape input_data for LSTM model
        X = np.array(input_df).reshape(1, input_df.shape[1], 1)
        print("Input data preprocessed successfully.")

        print("Making prediction...")
        prediction = model.predict(X)
        print(f"Prediction: {prediction}")

        result = 'Churn' if prediction[0][0] > 0.5 else 'Not Churn'
        print(f"Prediction result: {result}")

        return result

    except Exception as e:
        print(f'Error during prediction: {e}')
        return None

# Fungsi untuk mendapatkan peran pengguna dari database
def get_user_role(username):
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT role FROM tbl_user WHERE username = %s", (username,))
    role = cursor.fetchone()[0]
    cursor.close()
    return role

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        user_role = get_user_role(session['username'])
        if user_role == 'admin':
            return render_template('dashboard.html', username=session['username'], role=user_role)
        elif user_role == 'pegawai':
            return render_template('dashboard-user.html', username=session['username'], role=user_role)
        
        try:
            cur = mysql.connection.cursor()
            cur.execute('SELECT COUNT(*) FROM tbl_testing')
            uji = cur.fetchone()  # Mengambil nilai COUNT(*)
            cur.execute('SELECT COUNT(*) FROM tbl_training')
            latih = cur.fetchone()  # Mengambil nilai COUNT(*)
            cur.execute('SELECT nama FROM tbl_user')
            nama = cur.fetchone()  # Mengambil nilai COUNT(*)
            cur.close()
            
            return render_template('dashboard.html', username=session['username'], datatesting=uji[0], datatraining=latih[0], nama=nama)
        
        except Exception as e:
            return render_template('dashboard.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Cek login dari database
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM tbl_user WHERE username = %s AND password = %s", (username, password))
        user = cursor.fetchone()
        cursor.close()
        
        if user:
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            flash('Login failed. Please check your username and password.')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# Fungsi untuk memeriksa apakah file yang diunggah diizinkan
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv'}

@app.route('/train', methods=['GET', 'POST'])
def train():
    if 'username' not in session:
        return redirect(url_for('login'))

    user_role = get_user_role(session['username'])
    if user_role != 'admin':
        flash('Akses tidak sah: Hanya admin yang dapat mengakses halaman ini.')
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        try:
            file = request.files['file']
            
            if file.filename == '':
                flash('Tidak ada file yang dipilih untuk diunggah.')
                return redirect(url_for('train'))
            
            # Periksa apakah file yang diunggah memiliki ekstensi yang diizinkan
            if file and allowed_file(file.filename):
                col_name = ['Nama', 'Service_types', 'Packet_service', 'Media_transmisi', 'Bandwidth', 'State', 'Partner', 'Type_contract', 'Complaint', 'Churn']
                data = pd.read_csv(file, names=col_name, header=0, sep=',', skipinitialspace=True)
                
                # Cek dan mengganti nilai nan dengan None
                data.replace({np.nan: None}, inplace=True)
                
                cursor = mysql.connection.cursor()
                added_rows = 0  # Counter for added rows
                skipped_rows = 0  # Counter for skipped rows (duplicates)
                
                for index, row in data.iterrows():
                    # Cek apakah data sudah ada di database
                    cursor.execute("""
                        SELECT * FROM tbl_training 
                        WHERE Nama = %s AND Service_types = %s AND Packet_service = %s AND Media_transmisi = %s 
                        AND Bandwidth = %s AND State = %s AND Partner = %s AND Type_contract = %s AND Complaint = %s AND Churn = %s
                    """, (row['Nama'], row['Service_types'], row['Packet_service'], row['Media_transmisi'], row['Bandwidth'], row['State'], row['Partner'], row['Type_contract'], row['Complaint'], row['Churn']))
                    
                    existing_row = cursor.fetchone()
                    
                    if existing_row:
                        skipped_rows += 1
                        continue  # Skip this row
                    
                    # Simpan data training ke database jika tidak duplikat
                    cursor.execute("""
                        INSERT INTO tbl_training 
                        (Nama, Service_types, Packet_service, Media_transmisi, Bandwidth, State, Partner, Type_contract, Complaint, Churn) 
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (row['Nama'], row['Service_types'], row['Packet_service'], row['Media_transmisi'], row['Bandwidth'], row['State'], row['Partner'], row['Type_contract'], row['Complaint'], row['Churn']))
                    added_rows += 1

                mysql.connection.commit()
                cursor.close()
                
                # Latih model jika ada data baru yang ditambahkan
                if added_rows > 0:
                    train_model(data)
                    flash(f'Data training berhasil diunggah dan model dilatih. {added_rows} baris ditambahkan, {skipped_rows} baris dilewati karena duplikat.')
                else:
                    train_model(data)
                    flash(f'model dilatih.')
                    flash(f'Tidak ada data baru yang ditambahkan. {skipped_rows} baris dilewati karena duplikat.')
                return redirect(url_for('train'))
        
        except Exception as e:
            flash(f'Terjadi kesalahan: {str(e)}')
            return redirect(url_for('train'))
        
    if request.method == 'GET':
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM tbl_training")
        history = cursor.fetchall()
        cursor.close()
        #return render_template('training-prediksi.html', tbl_datatraining = history)

    return render_template('training-prediksi.html',  tbl_datatraining = history)

@app.route('/deletedatatraining/<string:id_data>', methods = ['POST','DELETE'])
def deletedatatraining(id_data):
    if (request.form['_method'] == 'DELETE'):
        flash("Delete Data Training Berhasil")
        cur = mysql.connection.cursor()
        cur.execute("DELETE FROM tbl_training WHERE id_training = %s", (id_data,))
        mysql.connection.commit()
        cur.close()
        return redirect (url_for('datatraining'))    


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))

    result = None  # Inisialisasi result dengan None

    if request.method == 'POST':
        try:
            # Ambil input dari form
            input_data = {
                'Service_types': request.form['Service_types'],
                'Packet_service': request.form['Packet_service'],
                'Media_transmisi': request.form['Media_transmisi'],
                'Bandwidth': float(request.form['Bandwidth']),
                'State': request.form['State'],
                'Partner': request.form['Partner'],
                'Type_contract': request.form['Type_contract'],
                'Complaint': request.form['Complaint']
            }
            #flash(f'{input_data}')
            # input_data = pd.DataFrame([input_data])
            #flash(f'{input_data}')
            # Prediksi menggunakan fungsi predict_model
            result = predict_model(input_data)

            if result is None:
                flash('Prediksi gagal. Mohon coba lagi.')
            else:
                # Simpan hasil prediksi ke database tbl_prediksi
                cursor = mysql.connection.cursor()
                cursor.execute("""
                    INSERT INTO tbl_prediksi 
                    (Nama, Service_types, Packet_service, Media_transmisi, Bandwidth, State, Partner, Type_contract, Complaint, Prediction) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (request.form['Nama'], input_data['Service_types'], input_data['Packet_service'], input_data['Media_transmisi'], input_data['Bandwidth'], input_data['State'], input_data['Partner'], input_data['Type_contract'], input_data['Complaint'], result))
                mysql.connection.commit()
                cursor.close()

        except Exception as e:
            flash(f'Terjadi kesalahan: {str(e)}')

    return render_template('input-prediksi.html', result=result)

@app.route('/training_history', )
def training_history():
    if 'username' not in session:
        return redirect(url_for('login'))

    #user_role = get_user_role(session['username'])
    #if user_role != 'admin':
    #    flash('Akses tidak sah: Hanya admin yang dapat mengakses halaman ini.')
    #    return redirect(url_for('dashboard'))

    cursor = mysql.connection.cursor()
    cursor.execute("SELECT * FROM tbl_training")
    history = cursor.fetchall()
    cursor.close()
    return render_template('training-prediksi.html', tbl_datatraining = history)

@app.route('/prediction_history')
def prediction_history():
    if 'username' not in session:
        return redirect(url_for('login'))

    cursor = mysql.connection.cursor()
    cursor.execute("SELECT * FROM tbl_prediksi")
    history = cursor.fetchall()
    cursor.close()
    return render_template('history-prediksi.html', tbl_dataprediksi=history)

#def create_hashed_password(password):
#    return bcrypt.generate_password_hash(password).decode('utf-8')

# @app.route('/insert', methods=['POST'])
# def insert():
#     if request.method == "POST":
#         flash("Register Berhasil")
#         nama = request.form['nama']
#         email = request.form['email']
#         username = request.form['username']
#         password = request.form['password']
#         role = request.form['role']
    
#         #hashed_password = create_hashed_password(password)
        
#         cur = mysql.connection.cursor()
#         cur.execute("INSERT INTO tbl_user (nama, email, username, password, role) VALUES (%s, %s, %s, %s, %s)", (nama, email, username, password, level))
#         mysql.connection.commit()
#         cur.close()
#         return redirect(url_for('user'))
    
@app.route('/delete/<string:id_data>', methods = ['POST','DELETE'])
def delateuser(id_data):
    if (request.form['_method'] == 'DELETE'):
        flash("Delete Data Berhasil")
        cur = mysql.connection.cursor()
        cur.execute("DELETE FROM tbl_user WHERE id_user = %s", (id_data,))
        mysql.connection.commit()
        cur.close()
        return redirect (url_for('user'))

@app.route('/user/reset/<user_id>', methods=['POST', 'PUT'])
def putUserReset(user_id):
    password = request.form['password']
    password2 = request.form['password2']
    
    if password == password2:
        #hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        hashed_password = password2
        flash("Change Password Berhasil")
        cur = mysql.connection.cursor()
        cur.execute("""UPDATE tbl_user SET password = %s WHERE id_user = %s""", (hashed_password, user_id)) 
        mysql.connection.commit()
        cur.close()
        return redirect(url_for('user'))
    else:
        flash("Password tidak cocok")
        return redirect(url_for('user'))

@app.route ('/user', methods=['GET', 'POST'])
def user ():
    if 'username' in session:
        return redirect(url_for('login'))
    
    if request.method == "POST":
        flash("Register Berhasil")
        nama = request.form['nama']
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
    
        #hashed_password = create_hashed_password(password)
        
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO tbl_user (nama, email, username, password, role) VALUES (%s, %s, %s, %s, %s)", (nama, email, username, password, level))
        mysql.connection.commit()
        cur.close()

    if request.method == 'GET':
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM tbl_training")
        history = cursor.fetchall()
        cursor.close()
        
        return render_template('user.html', tbl_user = data, username=session['username'] )
    else:
        return render_template('login.html')

def plot_churn_by_variable(df, variable):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=variable, hue='Churn')
    plt.title(f'Churn by {variable}')
    
    # Convert plot to PNG image
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    return img_base64

@app.route('/report')
def report():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    try:
        # Mendapatkan data dari database
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM tbl_training")
        training_data = cursor.fetchall()
        training_df = pd.DataFrame(training_data, columns=['id', 'Nama', 'Service_types', 'Packet_service', 'Media_transmisi', 'Bandwidth', 'State', 'Partner', 'Type_contract', 'Complaint', 'Churn'])
        
        cursor.execute("SELECT * FROM tbl_prediksi")
        prediction_data = cursor.fetchall()
        prediction_df = pd.DataFrame(prediction_data, columns=['id', 'Nama', 'Service_types', 'Packet_service', 'Media_transmisi', 'Bandwidth', 'State', 'Partner', 'Type_contract', 'Complaint', 'Prediction'])
        
        cursor.close()
        
        # Plot churn by variables
        churn_by_service_type = plot_churn_by_variable(training_df, 'Service_types')
        churn_by_packet_service = plot_churn_by_variable(training_df, 'Packet_service')
        churn_by_media_transmisi = plot_churn_by_variable(training_df, 'Media_transmisi')
        
        # Hitung jumlah churn dan non-churn dari data prediksi
        total_churn = prediction_df[prediction_df['Prediction'] == 'Churn'].shape[0]
        total_no_churn = prediction_df[prediction_df['Prediction'] == 'Not Churn'].shape[0]
        
        return render_template('report.html', 
                               churn_by_service_type=churn_by_service_type,
                               churn_by_packet_service=churn_by_packet_service,
                               churn_by_media_transmisi=churn_by_media_transmisi,
                               total_churn=total_churn,
                               total_no_churn=total_no_churn)
        
    except Exception as e:
        flash(f'Terjadi kesalahan: {str(e)}')
        return redirect(url_for('report.html'))

if __name__ == '__main__':
    app.run(debug=True)
