import pandas as pd
import matplotlib.pyplot as plt
import json

class LinearRegressionTraining:
	def __init__(self, file_path):
		self.data = pd.read_csv(file_path)
		self.theta0 = 0.0
		self.theta1 = 0.0
		self.normalize_data = self.normalizeData()
		self.data_x_mean = self.data.iloc[:, 0].mean()
		self.data_x_std = self.data.iloc[:, 0].std()
		self.data_y_mean = self.data.iloc[:, 1].mean()
		self.data_y_std = self.data.iloc[:, 1].std()

	def normalizeData(self): # Z-score normalization (μ=0, σ=1)
		data_scaled = self.data.copy()
		for column in data_scaled.columns:
			mean = data_scaled[column].mean()
			std = data_scaled[column].std()
			data_scaled[column] = (data_scaled[column] - mean) / std
		return data_scaled
	
	def hypothesisFunction(self, x):
		return self.theta0 + ( self.theta1 * x )
	
	def costFunction(self): # Mean Squared Error (MSE)
		m = len(self.normalize_data)
		total_errors = 0.0
		for x, y in self.normalize_data.itertuples(index=False):
			total_errors += ( self.hypothesisFunction(x) - y ) ** 2
		final_cost = total_errors / (2 * m)
		return final_cost
	
	def gradientDescent(self, alpha):
		m = len(self.normalize_data)
		global theta0, theta1
		total_errors0 = 0.0
		total_errors1 = 0.0
		for x, y in self.normalize_data.itertuples(index=False):
			error = self.hypothesisFunction(x) - y
			total_errors0 += error
			total_errors1 += error * x
		tmp_theta0 = (alpha / m) * total_errors0
		tmp_theta1 = (alpha / m) * total_errors1
		return tmp_theta0, tmp_theta1
	
	def train(self, alpha, config_x_value):
		cost = self.costFunction()
		exCurveData = []

		for i in range(10000):
			tmp_theta0, tmp_theta1 = self.gradientDescent(alpha=alpha)
			self.theta0 -= tmp_theta0
			self.theta1 -= tmp_theta1

			exCurveData.append(self.hypothesisFunction((config_x_value - self.data_x_mean) / self.data_x_std) * self.data_y_std + self.data_y_mean)
			new_cost = self.costFunction()
			if cost - new_cost < 1e-6:
				iterations = i
				break
			cost = new_cost

		real_rmse = (2 * cost) ** 0.5 * self.data_y_std
		self.save_model('model_weights.json')
		print(f"Training completed. Final cost: {cost:.6f}, RMSE: {real_rmse:.6f}")
		param_cost = {'theta0': self.theta0, 'theta1': self.theta1, 'rmse': real_rmse, 'iterations': iterations, 'exCurveData': exCurveData}
		return param_cost

	def save_model(self, filename):
		model = {
			'theta0': self.theta0,
			'theta1': self.theta1,
			'x_mean': self.data_x_mean,
			'x_std': self.data_x_std,
			'y_mean': self.data_y_mean,
			'y_std': self.data_y_std
		}
		with open(filename, 'w') as file:
			json.dump(model, file, indent=4)

	def plotData(self, config):
		plt.figure()
		x_values = self.data.iloc[:, 0]
		y_values = self.data.iloc[:, 1]
		plt.scatter(x_values, y_values, color='blue', label='Data Points')

		x_range = pd.Series([x_values.min(), x_values.max()])
		normalized_x_range = (x_range - self.data.iloc[:, 0].mean()) / self.data.iloc[:, 0].std()
		y_range = self.hypothesisFunction(normalized_x_range) * self.data_y_std + self.data_y_mean
		plt.plot(x_range, y_range, color='red', label='Regression Line')

		plt.xlabel(config['plot_config']['x_label'])
		plt.ylabel(config['plot_config']['y_label'])
		plt.title(config['plot_config']['title'])
		plt.legend()
		plt.show()

if __name__ == "__main__":
	trainer = LinearRegressionTraining('car_mileage.csv')
	trainer.train(0.1, 100000)