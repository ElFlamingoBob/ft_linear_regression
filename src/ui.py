import tkinter as tk
from tkinter import messagebox, ttk
import matplotlib.pyplot as plt
import json

from train import LinearRegressionTraining
from predict import LinearRegressionPredictor

def clear_window(root):
	for widget in root.winfo_children():
		widget.destroy()

def load_config(filename):
	with open(filename, 'r') as file:
		config = json.load(file)
	return config

def plotTest(config, exCurveData):
	plt.figure()
	plt.plot(range(len(exCurveData)), exCurveData, color='green', label='Ex Curve')
	plt.xlabel("Iterations")
	plt.ylabel(config["plot_config"]["y_label"])
	plt.title(config["plot_config"]["ex_curve_title"])
	plt.legend()
	plt.show()

def param_button_clicked(param_cost):
	popup = tk.Toplevel()
	popup.title("Parameters and RMSE")
	popup.geometry("500x150")
	theta0_text = f"θ0: {param_cost['theta0']}"
	theta1_text = f"θ1: {param_cost['theta1']}"
	final_cost_text = f"RMSE: {param_cost['rmse']}"
	iterations_text = f"Iterations: {param_cost['iterations']}"
	tk.Label(popup, text=theta0_text, justify=tk.CENTER, font=("Arial", 12, "bold")).pack(padx=20, pady=5, ipady=5)
	tk.Label(popup, text=theta1_text, justify=tk.CENTER, font=("Arial", 12, "bold")).pack(padx=20, pady=5)
	tk.Label(popup, text=final_cost_text, justify=tk.CENTER, font=("Arial", 12, "bold")).pack(padx=20, pady=5)
	tk.Label(popup, text=iterations_text, justify=tk.CENTER, font=("Arial", 12, "bold")).pack(padx=20, pady=5)

def training_phase(root, alpha, config, trainer):
	param_cost = trainer.train(alpha=alpha, config_x_value=config['ex_curve_x_value'])
	clear_window(root)
	tk.Button(root, text="Display Plot", command=lambda: trainer.plotData(config)).grid(row=0, column=0, pady=5)
	tk.Button(root, text="Display Parameters and RMSE", command=lambda: param_button_clicked(param_cost)).grid(row=0, column=1, pady=5)
	tk.Button(root, text="Ex Curve", command =lambda: plotTest(config, param_cost['exCurveData'])).grid(row=0, column=2, pady=5)
	tk.Label(root, text=config['entry_label']).grid(row=1, column=0)
	input = tk.Entry(root)
	input.grid(row=1, column=1)
	tk.Button(root, text=config['predict_button_text'], command=lambda: predict_button_clicked(input.get(), input, config)).grid(row=1, column=2)

def predict_button_clicked(input, entry_widget, config):
	predictor = LinearRegressionPredictor()
	try:
		predicted_price = predictor.predict(float(input))
		messagebox.showinfo(config['popup_prediction_title'], config['popup_prediction_message'].format(x=input, y=predicted_price))
	except ValueError:
		messagebox.showerror(config['popup_error_title'], config['popup_error_message'])
	entry_widget.delete(0, tk.END)


def setup_preConfig(root, filename):
	
	clear_window(root)
	pre_config = load_config('config.json')
	config = pre_config[filename]
	root.title(config['root_title'])
	tk.Label(root, text="Learning Rate (Alpha):").pack(padx=20, pady=(20, 0))
	alpha_scale = tk.Scale(root, from_=0.01, to=1.0, resolution=0.01, orient=tk.HORIZONTAL)
	alpha_scale.set(0.1)
	alpha_scale.pack(padx=20, pady=5)
	trainer = LinearRegressionTraining(filename)
	tk.Button(root, text="Start Training", command=lambda: training_phase(root, alpha_scale.get(), config, trainer)).pack(padx=20, pady=5)


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


if __name__ == "__main__":
	setup_gui()