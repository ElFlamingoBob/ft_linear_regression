import tkinter as tk
from tkinter import messagebox, ttk
import pandas as pd
import matplotlib.pyplot as plt
import json

x_subject = ""
y_subject = ""

def load_config(filename):
	with open(filename, 'r') as file:
		config = json.load(file)
	return config

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

data = pd.DataFrame()
data_scaled = pd.DataFrame()
mean_price = 0.0
std_price = 0.0
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

def gradientDescent(alpha):
	m = len(data_scaled)
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

def plotData(config):
	x_values = data[x_subject]
	y_values = data[y_subject]
	plt.scatter(x_values, y_values, color='blue', label='Data Points')

	x_range = pd.Series([x_values.min(), x_values.max()])
	normalized_x_range = (x_range - data[x_subject].mean()) / data[x_subject].std()
	y_range = hypothesisFunction(normalized_x_range) * std_price + mean_price
	plt.plot(x_range, y_range, color='red', label='Regression Line')

	plt.xlabel(config['plot_config']['x_label'])
	plt.ylabel(config['plot_config']['y_label'])
	plt.title(config['plot_config']['title'])
	plt.legend()
	plt.show()

def predict_button_clicked(input, entry_widget, config):
	try:
		predicted_price = handleInput(input)
		messagebox.showinfo(config['popup_prediction_title'], config['popup_prediction_message'].format(x=input, y=predicted_price))
	except ValueError:
		messagebox.showerror(config['popup_error_title'], config['popup_error_message'])
	entry_widget.delete(0, tk.END)

def plot_button_clicked(config):
	plotData(config)

def clear_window(root):
	for widget in root.winfo_children():
		widget.destroy()

def param_button_clicked(param_cost):
	popup = tk.Toplevel()
	popup.title("Parameters and Cost")
	popup.geometry("500x150")
	theta0_text = f"θ0: {param_cost['theta0']}"
	theta1_text = f"θ1: {param_cost['theta1']}"
	final_cost_text = f"Final Cost: {param_cost['cost']}"
	iterations_text = f"Iterations: {param_cost['iterations']}"
	tk.Label(popup, text=theta0_text, justify=tk.CENTER, font=("Arial", 12, "bold")).pack(padx=20, pady=5, ipady=5)
	tk.Label(popup, text=theta1_text, justify=tk.CENTER, font=("Arial", 12, "bold")).pack(padx=20, pady=5)
	tk.Label(popup, text=final_cost_text, justify=tk.CENTER, font=("Arial", 12, "bold")).pack(padx=20, pady=5)
	tk.Label(popup, text=iterations_text, justify=tk.CENTER, font=("Arial", 12, "bold")).pack(padx=20, pady=5)


def display_post_training_window(root, config, param_cost):
	clear_window(root)
	tk.Button(root, text="Display Plot", command=lambda: plot_button_clicked(config)).grid(row=0, column=0, pady=5)
	tk.Button(root, text="Display Parameters and Cost", command=lambda: param_button_clicked(param_cost)).grid(row=0, column=1, pady=5)
	tk.Label(root, text=config['entry_label']).grid(row=1, column=0)
	input = tk.Entry(root)
	input.grid(row=1, column=1)
	tk.Button(root, text=config['predict_button_text'], command=lambda: predict_button_clicked(input.get(), input, config)).grid(row=1, column=2)


def start_training(alpha, root, config):
	iterations = 0
	cost = costFunction()

	for i in range(10000) or cost == costFunction():
		gradientDescent(alpha=alpha)
		new_cost = costFunction()
		if new_cost == cost:
			iterations = i
			break
		cost = new_cost
	param_cost = {'theta0': theta0, 'theta1': theta1, 'cost': cost, 'iterations': iterations}
	print(theta0, theta1, cost, iterations)
	display_post_training_window(root, config, param_cost)
	

def setup_preConfig(root, filename):
	global data, data_scaled, mean_price, std_price
	
	clear_window(root)
	pre_config = load_config('config.json')
	config = pre_config[filename]
	data = pd.DataFrame(parseData(filename), columns=[x_subject, y_subject])
	data_scaled = normalizeData(data)
	mean_price = data[y_subject].mean()
	std_price = data[y_subject].std()
	root.title(config['root_title'])
	tk.Label(root, text="Learning Rate (Alpha):").pack(padx=20, pady=(20, 0))
	alpha_scale = tk.Scale(root, from_=0.01, to=1.0, resolution=0.01, orient=tk.HORIZONTAL)
	alpha_scale.set(0.1)
	alpha_scale.pack(padx=20, pady=5)
	tk.Button(root, text="Start Training", command=lambda: start_training(alpha_scale.get(), root=root, config=config)).pack(padx=20, pady=5)

def choose_file(root):
	tk.Label(root, text="Choose a file:").pack(padx=20, pady=5)
	combobox = ttk.Combobox(root, values=["paris_housing.csv", "car_mileage.csv"], state="readonly")
	combobox.pack(padx=20, pady=5)
	combobox.set("paris_housing.csv")
	tk.Button(root, text="Next", command=lambda: setup_preConfig(root=root, filename=combobox.get())).pack(padx=20, pady=5)

def setup_gui():
	root = tk.Tk()
	root.title("Predictor - Training Configuration")
	choose_file(root)
	root.mainloop()

def main():
	setup_gui()

if __name__ == "__main__":
	main()

# 100 iterations :  50 sqr =   557 427	| 1000 iterations :  50 sqr =   557 405
# 100 iterations : 100 sqr = 1 080 819	| 1000 iterations : 100 sqr = 1 080 812
# 100 iterations : 250 sqr = 2 650 996	| 1000 iterations : 250 sqr = 2 651 031


