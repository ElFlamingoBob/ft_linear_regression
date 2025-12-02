import tkinter as tk
from tkinter import simpledialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt

x_subject = ""
y_subject = ""


def normalizeData(data):
	data_scaled = data.copy()
	for column in data_scaled.columns:
		mean = data_scaled[column].mean()
		std = data_scaled[column].std()
		data_scaled[column] = (data_scaled[column] - mean) / std
	return data_scaled

def parseData(filename):
	data = []
	with open(filename, 'r') as file:
		for i, line in enumerate(file):
			if i == 0:
				global x_subject, y_subject
				x_subject, y_subject = map(str, line.strip().split(','))
				continue
			x, y = map(float, line.strip().split(','))
			data.append((x, y))
	return data

data = pd.DataFrame(parseData('paris_housing.csv'), columns=[x_subject, y_subject])
data_scaled = normalizeData(data)
mean_price = data[y_subject].mean()
std_price = data[y_subject].std()
# print(data)
theta0 = 0.0
theta1 = 0.0

def hypothesisFunction(x):
	return theta0 + ( theta1 * x )

def costFunction():
	m = len(data_scaled)
	total_errors = 0.0
	for x, y in data_scaled.itertuples(index=False):
		total_errors += ( hypothesisFunction(x) - y ) ** 2
	final_cost = total_errors / (2 * m)
	return final_cost


def gradientDescent():
	m = len(data_scaled)
	alpha = 0.1
	global theta0, theta1
	total_errors0 = 0.0
	total_errors1 = 0.0
	for x, y in data_scaled.itertuples(index=False):
		error = hypothesisFunction(x) - y
		total_errors0 += error
		total_errors1 += error * x
	theta0 -= (alpha / m) * total_errors0
	theta1 -= (alpha / m) * total_errors1

def handleInput(input_km):
	km_value = float(input_km)
	normalized_km = (km_value - data[x_subject].mean()) / data[x_subject].std()
	predicted_price = round((hypothesisFunction(normalized_km) * std_price) + mean_price)
	if predicted_price < 0:
		predicted_price = 37
	return predicted_price

def plotData():
	x_values = data[x_subject]
	y_values = data[y_subject]
	plt.scatter(x_values, y_values, color='blue', label='Data Points')

	x_range = pd.Series([x_values.min(), x_values.max()])
	normalized_x_range = (x_range - data[x_subject].mean()) / data[x_subject].std()
	y_range = hypothesisFunction(normalized_x_range) * std_price + mean_price
	plt.plot(x_range, y_range, color='red', label='Regression Line')

	plt.xlabel(x_subject)
	plt.ylabel(y_subject)
	plt.title(f'{y_subject} Prediction')
	plt.legend()
	plt.show()

def main():
	cost = costFunction()

	for i in range(1000):
		gradientDescent()
		cost = costFunction()
	print ("Theta0:", theta0)
	print ("Theta1:", theta1)
	print ("Final cost:", cost)
	

	plotData()
	root = tk.Tk()
	root.withdraw()

	text = simpledialog.askstring('Car Price Predictor', 'Enter km value to predict price:')
	while text is not None:
		try:
			predicted_price = handleInput(text)
			messagebox.showinfo('Prediction', f'Predicted price for {text} km is: {predicted_price} euros.')
			text = simpledialog.askstring('Car Price Predictor', 'Enter km value to predict price:')
		except ValueError:
			messagebox.showerror('Error', 'Please enter a valid number.')
			text = simpledialog.askstring('Car Price Predictor', 'Enter km value to predict price:')


if __name__ == "__main__":
	main()