import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

st.title('Tugas UAS Proyek Sains Data B ')
st.text("Isnain Fauziah 200411100007")
st.text("Nella Adrisia Hartono 200411100107")
data, preprocessing, modelling, implementasi = st.tabs(["Data","Preprocessing" ,"Modelling", "Implementasi"])


#data
def read_data():
    link = "https://raw.githubusercontent.com/nellaadrs/kelompokpro-1/main/SMGR.JK.csv"
    df = pd.read_csv(link, header="infer", index_col=False)
    return df
    
    
with data:
    st.write("""
	Dataset yang digunakan berasal dari data finance yahoo.com dari perusahaan PT Semen Indonesia (Persero) Tbk (SMGR.JK). Data ini berjumlah 247 Data dengan 7 fitur, yaitu: Date, Open, High, Low, Close,Adj, Close, Volume. Data yang digunakan ini merupakan catatan harga dalam kurun waktu 15 Juni 2022 - 15 Juni 2023. Dalam catatan data tersebut dapat dilakukan perhitungan tingkat kesalahan dalam melakukan prediksi beberapa hari yang akan datang .
	
	1. Date (Tanggal): Tanggal dalam data time series mengacu pada tanggal tertentu saat data keuangan dikumpulkan atau dilaporkan. Ini adalah waktu kapan data keuangan yang terkait dengan PT Semen Indonesia Indonesia dicatat.
	2. Open (Harga Pembukaan): Harga pembukaan adalah harga perdagangan PT Adaro Minerals Indonesia pada awal periode waktu tertentu, seperti hari perdagangan atau sesi perdagangan. Harga pembukaan menunjukkan harga perdagangan pertama dari PT Adaro Minerals Indonesia pada periode tersebut.
	3. High (Harga Tertinggi): Harga tertinggi adalah harga tertinggi yang dicapai oleh PT Adaro Minerals Indonesia selama periode waktu tertentu, seperti hari perdagangan atau sesi perdagangan. Harga tertinggi mencerminkan harga perdagangan tertinggi yang dicapai oleh PT Adaro Minerals Indonesia dalam periode tersebut.
	4. Low (Harga Terendah): Harga terendah adalah harga terendah yang dicapai oleh PT Adaro Minerals Indonesia selama periode waktu tertentu, seperti hari perdagangan atau sesi perdagangan. Harga terendah mencerminkan harga perdagangan terendah yang dicapai oleh PT Adaro Minerals Indonesia dalam periode tersebut.
	5. Close (Harga Penutupan): Harga penutupan adalah harga terakhir dari PT Adaro Minerals Indonesia pada akhir periode waktu tertentu, seperti hari perdagangan atau sesi perdagangan. Harga penutupan menunjukkan harga terakhir di mana PT Adaro Minerals Indonesia diperdagangkan pada periode tersebut.
	6. Adj Close (Harga Penutupan yang Disesuaikan): Adj Close, atau harga penutupan yang disesuaikan, adalah harga penutupan yang telah disesuaikan untuk faktor-faktor seperti dividen, pemecahan saham, atau perubahan lainnya yang mempengaruhi harga saham PT Adaro Minerals Indonesia. Ini memberikan gambaran yang lebih akurat tentang kinerja saham dari waktu ke waktu karena menghilangkan efek dari perubahan-perubahan tersebut.
	7. Volume: Volume dalam konteks data keuangan PT Adaro Minerals Indonesia mengacu pada jumlah saham yang diperdagangkan selama periode waktu tertentu, seperti hari perdagangan atau sesi perdagangan. Volume mencerminkan seberapa aktifnya perdagangan saham PT Adaro Minerals Indonesia dalam periode tersebut.
    
	"""
	)
    df=read_data()
    df


with preprocessing:
		st.write("Data yang telah di preprocessing :")
		training_set = df.iloc[:197, 1:-1].values
		test_set = df.iloc[49:, 1:-1].values
    # Feature Scaling
		sc = MinMaxScaler(feature_range = (0, 1))
		training_set_scaled = sc.fit_transform(training_set)
			# Creating a data structure with 60 time-steps and 1 output
		X_train = []
		y_train = []
		x = 5
		for i in range(x, 197):
			X_train.append(training_set_scaled[i-x:i, 0])
			y_train.append(training_set_scaled[i, 0])
		X_train, y_train = np.array(X_train), np.array(y_train)
		X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
		#(740, 60, 1)
		xtrainbaru = np.reshape(X_train, (192, 5))
		st.write(xtrainbaru)
	


with  modelling:
	with st.form("modeling"):
		st.subheader('Modeling')
		st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
		# naive = st.checkbox('Gaussian Naive Bayes')
		k_nn = st.checkbox('K-Nearest Neighbors')
		svr = st.checkbox('Support Vector Regression')
		# destree = st.checkbox('Decision Tree')
		submitted = st.form_submit_button("Submit")

	if k_nn:
		# import knn
		from sklearn.neighbors import KNeighborsRegressor
		neigh = KNeighborsRegressor(n_neighbors=3)
		modelknn=neigh.fit(xtrainbaru, y_train)
		# Definisikan dataset_train dan dataset_test
		dataset_train = df.iloc[:197, 1:2]  # Menggunakan 45 baris pertama sebagai data latihan
		dataset_test = df.iloc[49:, 1:2]  # Menggunakan baris setelah 45 sebagai data uji

		# Menggabungkan dataset_train dan dataset_test
		dataset_total = pd.concat([dataset_train, dataset_test], axis=0)

		# Mengambil input dari dataset_total
		inputs = dataset_total[len(dataset_total) - len(dataset_test) - 5:].values
		inputs = inputs.reshape(-1,1)
		inputs =sc.fit_transform(inputs)
		X_test = []
		for i in range(5, 24):
			X_test.append(inputs[i-5:i, 0])
		X_test = np.array(X_test)
		X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
		# print(X_test.shape)

		xtestbaru = np.reshape(X_test, (19, 5))

		predicted_pass = modelknn.predict(xtestbaru)

		predicted_pass = predicted_pass.reshape(-1,1)
		prediksi= sc.inverse_transform(predicted_pass)
		dataset_test=dataset_test.iloc[0:19]
    
		from sklearn.metrics import mean_absolute_percentage_error

		hasil_mape = mean_absolute_percentage_error(dataset_test, prediksi)
		st.write('MAPE Model K-Nearest Neighbors :', hasil_mape)
	if svr:
		from sklearn.svm import SVR
		svm = SVR(kernel='rbf')
		model=svm.fit(xtrainbaru, y_train)
		# import knn
		# from sklearn.neighbors import KNeighborsRegressor
		# neigh = KNeighborsRegressor(n_neighbors=3)
		# modelknn=neigh.fit(xtrainbaru, y_train)
		# Definisikan dataset_train dan dataset_test
		dataset_train = df.iloc[:197, 1:2]  # Menggunakan 45 baris pertama sebagai data latihan
		dataset_test = df.iloc[49:, 1:2]  # Menggunakan baris setelah 45 sebagai data uji

		# Menggabungkan dataset_train dan dataset_test
		dataset_total = pd.concat([dataset_train, dataset_test], axis=0)

		# Mengambil input dari dataset_total
		inputs = dataset_total[len(dataset_total) - len(dataset_test) - 5:].values
		inputs = inputs.reshape(-1,1)
		inputs =sc.fit_transform(inputs)
		X_test = []
		for i in range(5, 24):
			X_test.append(inputs[i-5:i, 0])
		X_test = np.array(X_test)
		X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
		# print(X_test.shape)

		xtestbaru = np.reshape(X_test, (19, 5))

		predicted_pass = model.predict(xtestbaru)

		predicted_pass = predicted_pass.reshape(-1,1)
		prediksi= sc.inverse_transform(predicted_pass)
		dataset_test=dataset_test.iloc[0:19]
    
		from sklearn.metrics import mean_absolute_percentage_error

		hasil_mape = mean_absolute_percentage_error(dataset_test, prediksi)
		st.write('MAPE Model Support Vector Regression :', hasil_mape)

	

with  implementasi:
    with st.form("my_form"):
        st.write("Implementasi")
        Open = st.number_input('Open')
        High = st.number_input('High')
        Low = st.number_input('Low')
        Close = st.number_input('Close')
        Adj_Close = st.number_input('Adj Close')
        submitted = st.form_submit_button("Submit")
        a = np.array([[Open,High, Low, Close, Adj_Close]])
        data_inputan = pd.DataFrame(a, columns= ["Open","High","Low", "Close", "Adj_Close"])
    if submitted:
	    st.write(mean_absolute_percentage_error(dataset_test, prediksi))

        




