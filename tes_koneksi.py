import mysql.connector 
from mysql.connector import Error 

def test_mysql_connection(host, database, user, password):
    try:
        connection = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )
        if connection.is_connected():
            print("Berhasil terhubung ke database MySQL")
    except Error as e:
        print(f"Error: {e}")
    finally:
        if connection.is_connected():
            connection.close()
            print("Koneksi MySQL ditutup")

# Ganti parameter di bawah dengan detail koneksi Anda
test_mysql_connection('localhost', 'prediksi_churn', 'root', '')
