
from flask import Flask,request,jsonify
from keras.preprocessing.sequence import TimeseriesGenerator
# from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

@app.route('/predict_model',methods=['GET'])
def predict_model():
	from datetime import datetime
	import pandas as pd
	import tensorflow as tf

	# ----- รับข้อมูลจากหน้า web ---------
	d = {}
	d['Soybean_meal_US'] = float(request.args['Soybean_meal_US'])
	d['Crude_Oil'] = float(request.args['Crude_Oil'])
	d['New_Month'] = float(request.args['New_Month'])
	d['Year'] = float(request.args['Year'])

	df_input = pd.DataFrame([d])  # แปลงเป็น dataframe
	# print(df_input.dtypes)
	# d['Soybean_meal_US'] = d['Soybean_meal_US'].astype(float)
	year = int(d['Year'])
	month = int(d['New_Month'])
	day = 1
	print('year')
	print(year - 543)

	if month == 1:
		start_date = datetime(year - 544, 10, day)  # 12 11 10
		end_date = datetime(year - 543, month, day)
	elif month == 2:
		start_date = datetime(year - 544, 11, day)  # 1 12 11
		end_date = datetime(year - 543, month, day)
	elif month == 3:
		start_date = datetime(year - 544, 12, day)  # 2 1 12
		end_date = datetime(year - 543, month, day)
	else:
		start_date = datetime(year - 543, month - 3, day)
		end_date = datetime(year - 543, month, day)
	#

	print(end_date)
	print(start_date)

	# ------- อ่านข้อมูล excel  ---------

	df_crude_oil_data = pd.read_excel('crude_oil_data.xls')  # soybean_meal_data.xls
	df_soybean_meal_data = pd.read_excel('soybean_meal_data.xls')  # soybean_meal_data.xls
	# ------- ทำ excel to array --------
	array_crude_oil_data = df_crude_oil_data.to_numpy()
	array_soybean_meal_data = df_soybean_meal_data.to_numpy()

	# -------- ดึงข้อมูลที่ต้องการจาก array ไปใส่อีก array ----------
	new_crude_oil = []
	for i in range(len(array_crude_oil_data)):
		new_crude_oil.append([])
		for j in range(1, 3):
			if array_crude_oil_data[i][1] >= start_date and array_crude_oil_data[i][1] < end_date:
				new_crude_oil[i].append(array_crude_oil_data[i][j])
				print(array_crude_oil_data[i][j])

	new_soybean_meal = []
	for i in range(len(array_soybean_meal_data)):
		new_soybean_meal.append([])
		for j in range(1, 3):
			if array_soybean_meal_data[i][1] >= start_date and array_soybean_meal_data[i][1] < end_date:
				new_soybean_meal[i].append(array_soybean_meal_data[i][j])
				print(array_soybean_meal_data[i][j])

	# ทำ new array to dataframe and drop all Nan or NaT
	soybean_meal_complete = pd.DataFrame(new_soybean_meal, columns=['Date', 'Soybean_meal_us'])
	soybean_meal_complete = soybean_meal_complete.dropna()
	crude_oil_complete = pd.DataFrame(new_crude_oil, columns=['Date', 'Crude_Oil'])
	crude_oil_complete = crude_oil_complete.dropna()

	print(soybean_meal_complete)
	print(crude_oil_complete)

	# แยก เดือน ปี
	soybean_meal_complete['New_Month'] = pd.to_datetime(soybean_meal_complete['Date']).dt.strftime('%m').astype('int')
	soybean_meal_complete['Year'] = pd.to_datetime(soybean_meal_complete['Date']).dt.strftime('%Y').astype('int')
	crude_oil_complete['New_Month'] = pd.to_datetime(crude_oil_complete['Date']).dt.strftime('%m').astype('int')
	crude_oil_complete['Year'] = pd.to_datetime(crude_oil_complete['Date']).dt.strftime('%Y').astype('int')
	soybean_meal_complete = soybean_meal_complete[['New_Month', 'Year', 'Soybean_meal_us']]
	crude_oil_complete = crude_oil_complete[['New_Month', 'Year', 'Crude_Oil']]

	print('newwwwwwwwwwwwwwwwwwwww')
	# print(crude_oil_complete)
	print(soybean_meal_complete)

	# --------- ทำ array check status เดือนที่ขาดของ crude oil and soybean ----------
	import numpy as np

	# array status
	check = []
	reverse = 12
	print('startttttttttttttttttttttt')
	print(month)

	for i in range(4):
		if (month <= 0):
			check.append((reverse, 0))
			reverse = reverse - 1
		else:
			check.append((month, 0))
		month = month - 1
		crude_oil_mat_vals = np.vstack(check)
		soybean_mat_vals = np.vstack(check)

	print('check stata')
	print(check)
	# new array
	crude_oil_mat_vals = np.delete(crude_oil_mat_vals, 0, 0)
	soybean_mat_vals = np.delete(soybean_mat_vals, 0, 0)

	# --------- check shape dataframe ของ Crud oil and Soybean meal ถ้าไม่ครบก็จะไปใส่สถานะ ใน array soybean_mat_vals / crude_oil_mat_vals
	# ของ SOYBEAN MEAL
	if soybean_meal_complete.shape == (3, 2):
		print('yes')
	else:
		# ใส่สถานะว่าเดือนไหนมีข้อมูล
		array_soybean_for_status = np.array(soybean_meal_complete)
		for i in range(len(soybean_mat_vals)):
			for j in range(len(array_soybean_for_status)):
				if soybean_mat_vals[i][0] == int(array_soybean_for_status[j][0]):
					print(soybean_mat_vals[i][0])
					soybean_mat_vals[i][1] = 1
	print('----------soybean_mat_vals-----------')
	print(soybean_mat_vals)

	# ของ CRUDE OIL
	if crude_oil_complete.shape == (3, 2):
		print('yes')
	else:
		# ใส่สถานะว่าเดือนไหนมีข้อมูล
		array_crude_oil_for_status = np.array(crude_oil_complete)
		for i in range(len(crude_oil_mat_vals)):
			for j in range(len(array_crude_oil_for_status)):
				if crude_oil_mat_vals[i][0] == int(array_crude_oil_for_status[j][0]):
					print(crude_oil_mat_vals[i][0])
					crude_oil_mat_vals[i][1] = 1

	print('----------crude_oil_mat_vals-----------')
	print(crude_oil_mat_vals)

	# ปริ้นอาเรย์เช็คสถานะออกมาดู count เอาไว้นับจำนวนเดือนที่ขาดเพื่อใช้ใน loop ถัดไป
	count_null_soybean = 0
	for i in range(len(soybean_mat_vals)):
		for j in range(2):
			if soybean_mat_vals[i][j] == 0:
				count_null_soybean += 1
	print('count_null_soybean')
	print(count_null_soybean)
	#
	count_null_crude_oil = 0
	for i in range(len(crude_oil_mat_vals)):
		for j in range(2):
			if crude_oil_mat_vals[i][j] == 0:
				count_null_crude_oil += 1
	print('count_null_crude_oil')
	print(count_null_crude_oil)

	# ใส่ null to crude oil dataframe and soybean meal สำหรับเดือนที่ขาด
	# soybean
	for i in range(len(soybean_mat_vals)):
		if soybean_mat_vals[i][1] == 0:
			if soybean_mat_vals[i][0] == 12:

				fill_Null = {'New_Month': soybean_mat_vals[i][0], 'Year': year - 544, 'Soybean_meal_us': [np.nan]}
				fill_Null_Data = pd.DataFrame(fill_Null)  # แปลงเป้น dataframe
				# รวม dataframe ข้อมูลที่มีกับข้อมูลที่ null เข้าด้วยกันแล้ว sort และเติมค่า mean
				# print('result_for_predict----')
				# print(result_for_predict)
				# print('------------')
				frames1 = [soybean_meal_complete, fill_Null_Data]

				# print('frames1')
				# print(frames1)
				# print('----')
				soybean_meal_complete = pd.concat(frames1)

				# print('result_for_predict1')
				# print(result_for_predict)
				# print('-------------')
				soybean_meal_complete = soybean_meal_complete.sort_values(by=['New_Month'])
				soybean_meal_complete = soybean_meal_complete.fillna(soybean_meal_complete.mean())
			#
			# print(fill_Null_Data)
			# print(result_for_predict)
			# print('----------------')
			#
			else:
				fill_Null_soy = {'New_Month': soybean_mat_vals[i][0], 'Year': year - 543, 'Soybean_meal_us': [np.nan]}
				fill_Null_Data_soy = pd.DataFrame(fill_Null_soy)  # แปลงเป้น dataframe
				# รวม dataframe ข้อมูลที่มีกับข้อมูลที่ null เข้าด้วยกันแล้ว sort และเติมค่า mean
				# print('result_for_predict----')
				# print(result_for_predict)
				# print('------------')
				frames_soy = [soybean_meal_complete, fill_Null_Data_soy]

				# print('frames1')
				# print(frames1)
				# print('----')
				soybean_meal_complete = pd.concat(frames_soy)

				# print('result_for_predict1')
				# print(result_for_predict)
				# print('-------------')
				soybean_meal_complete = soybean_meal_complete.sort_values(by=['New_Month'])
				soybean_meal_complete = soybean_meal_complete.fillna(soybean_meal_complete.mean())

		print('fill_Null_Data_soy')
		print('soybean_meal_complete')
		# print(fill_Null_Data_soy)
		print(soybean_meal_complete)
		print('----------------')

	# crude oil

	for i in range(len(crude_oil_mat_vals)):
		if crude_oil_mat_vals[i][1] == 0:
			if crude_oil_mat_vals[i][0] == 12:

				fill_Null_soy = {'New_Month': crude_oil_mat_vals[i][0], 'Year': year - 544, 'Crude_Oil': [np.nan]}
				fill_Null_Data_soy = pd.DataFrame(fill_Null_soy)  # แปลงเป้น dataframe
				# รวม dataframe ข้อมูลที่มีกับข้อมูลที่ null เข้าด้วยกันแล้ว sort และเติมค่า mean
				# print('result_for_predict----')
				# print(result_for_predict)
				# print('------------')
				frames_oil = [crude_oil_complete, fill_Null_Data_soy]

				# print('frames1')
				# print(frames1)
				# print('----')
				crude_oil_complete = pd.concat(frames_oil)

				# print('result_for_predict1')
				# print(result_for_predict)
				# print('-------------')
				crude_oil_complete = crude_oil_complete.sort_values(by=['New_Month'])
				crude_oil_complete = crude_oil_complete.fillna(crude_oil_complete.mean())
			#
			# print(fill_Null_Data)
			# print(result_for_predict)
			# print('----------------')
			#
			else:
				fill_Null_oil = {'New_Month': crude_oil_mat_vals[i][0], 'Year': year - 543, 'Crude_Oil': [np.nan]}
				fill_Null_Data_oil = pd.DataFrame(fill_Null_oil)  # แปลงเป้น dataframe
				# รวม dataframe ข้อมูลที่มีกับข้อมูลที่ null เข้าด้วยกันแล้ว sort และเติมค่า mean
				# print('result_for_predict----')
				# print(result_for_predict)
				# print('------------')
				frames_oil = [crude_oil_complete, fill_Null_Data_oil]

				# print('frames1')
				# print(frames1)
				# print('----')
				crude_oil_complete = pd.concat(frames_oil)

				# print('result_for_predict1')
				# print(result_for_predict)
				# print('-------------')
				crude_oil_complete = crude_oil_complete.sort_values(by=['New_Month'])
				crude_oil_complete = crude_oil_complete.fillna(crude_oil_complete.mean())

		print('fill_Null_Data_oil')
		print('crude_oil_complete')
		print(crude_oil_complete)
	# print(fill_Null_Data_oil)

	# เอา dataframe 2 อันมารวมกัน crude_oil_complete+soybean_meal_complete
	new_soybean_for_concat = pd.DataFrame(soybean_meal_complete, columns=['Soybean_meal_us'])
	new_soybean_for_concat = new_soybean_for_concat.reset_index(drop=True)
	crude_oil_complete = crude_oil_complete.reset_index(drop=True)
	data_from_excel_complete = pd.concat([crude_oil_complete, new_soybean_for_concat], axis=1)

	print('เอา dataframe 2 อันมารวมกัน crude_oil_complete+soybean_meal_complete')
	print(data_from_excel_complete)
	# ----------------------------------จัดการ data excel เสร็จแล้ว -------------------------------------------.

	# ข้อมูล Input user from User Interface
	user_input = [[d['New_Month'], year - 543, d['Crude_Oil'], d['Soybean_meal_US']]]
	print('#ข้อมูล Input user from User Interface')
	print(user_input)
	# เอา dataframe data_from_excel_complete ข้อมูลจากผู้ใช้ที่หน้า UI
	user_input_for_concat = pd.DataFrame(user_input, columns=['New_Month', 'Year', 'Crude_Oil', 'Soybean_meal_us'])
	user_input_for_concat = user_input_for_concat.reset_index(drop=True)
	data_from_excel_complete = data_from_excel_complete.reset_index(drop=True)
	frames_last = [data_from_excel_complete, user_input_for_concat]
	complete_all_data = pd.concat(frames_last)
	print('complete_all_data')
	print(complete_all_data)

	# ----------------------------------จัดการ data ทุกอย่างเรียบร้อย -------------------------------------------.

	# dataframe reslut for predict1 to array for predict
	list_for_predict = complete_all_data.values.tolist()
	list_for_predict
	array_for_predict = np.array(list_for_predict)
	print(array_for_predict.shape)  # shape (4,4)
	array_for_predict = array_for_predict.reshape((1, 4, 4))  # reshape (1,4,4) จะได้ทำนายได้

	# เอาอาเรย์ไปทำนาย
	model = tf.keras.models.load_model('Bidirectional_4_slide_mape4point5.h5', compile=False)
	model.compile(loss='mean_squared_error',
				  optimizer='adam', metrics=[tf.keras.metrics.MeanAbsolutePercentageError(),
											 tf.keras.metrics.MeanAbsoluteError(),
											 tf.keras.metrics.MeanSquaredError()])

	result_predict = model.predict(array_for_predict)
	# # print("ราคาที่ทำนายได้ของวันที่ " + end_date + " เท่ากับ " + str(result_predic[0][0]))
	num = float(result_predict[-1])
	print(type(num))
	predict = "{:.2f}".format(round(num, 2))
	print(predict)

	return jsonify(predict)


@app.route('/update_data',methods=['GET'])
def update_data():

	import pandas as pd
	import yfinance as yf
	from datetime import datetime

	# ----- รับข้อมูลจากหน้า web ---------
	d = {}
	d['New_Month'] = float(request.args['New_Month'])
	d['Year'] = float(request.args['Year'])
	d['priceThai'] = float(request.args['priceThai'])

	select_date = datetime(int(d['Year'])-543, int(d['New_Month']), 1)
	print(select_date)
	# df_input = pd.DataFrame([d]) #แปลงเป็น dataframe


	df = pd.read_excel('dataofPrice_TrainModel.xlsx')
	df['New_Month'] = pd.to_datetime(df['Date']).dt.strftime('%m').astype('int')
	df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y').astype('int')
	# print(df)

	last_day_df = df.tail(1)

	print('last day of')
	print(last_day_df)

	#ทำวันสำหรับใช้ดคงข้อมูล จากวันล่าสุุดของข้อมูลที่มี จนถึงปัจจุบัน
	start_date = datetime(last_day_df['Year'], last_day_df['New_Month'], 1)
	# print(start_date)

	current_dateTime = datetime.now()
	end_date = datetime(current_dateTime.year, current_dateTime.month, 1)
	# print(end_date)


	data_crude_oil = yf.download(tickers="CL=F", start=start_date, end=end_date, interval='1mo')
	data_crude_oil = data_crude_oil.reset_index()
	data_crude_oil['New_Month'] = pd.to_datetime(data_crude_oil['Date']).dt.strftime('%m').astype('int')
	data_crude_oil['Year'] = pd.to_datetime(data_crude_oil['Date']).dt.strftime('%Y').astype('int')
	data_crude_oil = pd.DataFrame(data_crude_oil, columns=['Date', 'New_Month', 'Year', 'Close'])
	# print('data crude oil')
	# print(data_crude_oil)

	data_soybean_meal = yf.download(tickers="ZM=F", start=start_date, end=end_date, interval='1mo')
	data_soybean_meal = data_soybean_meal.reset_index()
	data_soybean_meal = pd.DataFrame(data_soybean_meal, columns=['Close'])
	data_soybean_meal = data_soybean_meal.rename(columns={"Close": "Soybean_meal_US"})
	# print('data_soybean_meal')
	# print(data_soybean_meal)

	data_thai_import = pd.DataFrame()
	data_thai_import['Thai_Import'] = ['']
	new_data_from_yahoo = pd.DataFrame(data_crude_oil, columns=['Date', 'Thai_Import', 'Close'])
	new_data_from_yahoo = new_data_from_yahoo.rename(columns={"Close": "Crude_Oil"})

	new_data_from_yahoo = new_data_from_yahoo.reset_index(drop=True)
	data_soybean_meal = data_soybean_meal.reset_index(drop=True)
	new_data_from_yahoo = pd.concat([new_data_from_yahoo, data_soybean_meal], axis=1)
	new_data_from_yahoo = new_data_from_yahoo.rename(columns={"Close": "Soybean_meal_US"})
	# print('new_data_from_yahoo')
	# print(new_data_from_yahoo)

	new_data_from_yahoo.to_excel("new_data_from_yahoo.xlsx")
	df_from_excel = pd.read_excel('new_data_from_yahoo.xlsx') #soybean_meal_data.xls

	array_data_load = df_from_excel.to_numpy()

	# for i in range(len(array_data_load)):
	# 	for j in range(0, 2):
	# 		if array_data_load[i][j] == select_date:
	# 			df_from_excel.loc[[i, i], 'Thai_Import'] = d['priceThai']
	# 			print(i, i)
	# 			print(array_data_load[i][j])

	new_data_from_yahoo.to_excel("new_data_from_yahoo.xlsx")
	df_from_excel = pd.read_excel('new_data_from_yahoo.xlsx') #soybean_meal_data.xls

	print('read data from excel')
	print(df_from_excel)

	df_from_excel = df_from_excel.drop(['Unnamed: 0'], axis=1)
	print(df_from_excel)

	excel_base = pd.read_excel('dataofPrice_TrainModel.xlsx')

	finaly_df = pd.concat([excel_base, df_from_excel], ignore_index=True, sort=False)
	print(finaly_df.tail(5))

	finaly_df.to_excel("dataofPrice_TrainModel.xlsx")
	test_excel = pd.read_excel('dataofPrice_TrainModel.xlsx')  # soybean_meal_data.xls
	# test_excel = test_excel.drop(['Unnamed: 0'], axis=1)
	print(test_excel)
	return jsonify(1)

@app.route('/train_model',methods=['GET'])
def train_model():

	from tensorflow.keras.layers import Bidirectional
	import tensorflow as tf
	import pandas as pd

	from keras.models import load_model

	df = pd.read_excel('dataofPrice_TrainModel.xlsx')
	print(df)

	df['New_Month'] = pd.to_datetime(df['Date']).dt.strftime('%m').astype('int')
	df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y').astype('int')

	df_input_x = df[['New_Month', 'Year', 'Crude_Oil', 'Soybean_meal_US']]
	df_input_y = df[['Thai_Import']]
	df_input = df[['Soybean_meal_US']]

	x_list = df_input_x.values.tolist()
	y_list = df_input_y.values.tolist()

	# univariate lstm example
	from numpy import array
	from keras.models import Sequential
	from keras.layers import LSTM
	from keras.layers import Dense

	# split a univariate sequence into samples
	def split_sequence(sequence, n_steps):
		X = list()
		for i in range(len(sequence)):
			end_ix = i + n_steps
			if end_ix > len(sequence) - 1:
				break
			seq_x = sequence[i:end_ix]
			X.append(seq_x)
		return array(X)

	n_steps = 4
	x_train = split_sequence(x_list, n_steps)


	n_steps = 1
	y_train = split_sequence(y_list[3:], n_steps)
	y_train = y_train.flatten()

	split_time = 260
	x_train_split = x_train[:split_time]
	y_train_split = y_train[:split_time]

	x_test_split = x_train[split_time:]
	y_test_split = y_train[split_time:]

	n_features = 4
	# define model
	model = Sequential()
	model.add(Bidirectional(LSTM(128, activation='relu', return_sequences=True, ), input_shape=(4, 4)))
	# model.add(LSTM(96))
	model.add(Dense(1))
	# model.compile(optimizer='adam', loss='mse')
	model.compile(loss='mean_squared_error',
				  optimizer='adam', metrics=[tf.keras.metrics.MeanAbsolutePercentageError(),
											 tf.keras.metrics.MeanAbsoluteError(),
											 tf.keras.metrics.MeanSquaredError()])
	history = model.fit(x_train_split,
						y_train_split,
						#                     batch_size=32,
						epochs=1000,
						validation_data=(x_test_split, y_test_split),
						shuffle=False)

	model.save('py_test.h5')
	model = load_model('py_test.h5')
	score = model.evaluate(x_test_split, y_test_split, verbose=0)
	print("Loss is", score[0])
	print("Mean Absolute Percentage Error/ MAPE is", score[1])
	print("Mean Absolute Error/ MAE is", score[2])
	print("Mean Squared Error/ MSE is", score[3])


	return jsonify(0)

@app.route('/excel_value',methods=['GET'])
def read_excelrow():
	wan = "55555555555555555555555555555555555"
	print("wannarak444444444444444444444444444444444444444444")
	return jsonify(wan)

if __name__ == '__main__':
	app.run(debug=False)