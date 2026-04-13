import json

class LinearRegressionPredictor:
	def __init__(self):
		model = self.load_model('model_weights.json')

		self.theta0 = model['theta0']
		self.theta1 = model['theta1']
		self.x_mean = model['x_mean']
		self.x_std = model['x_std']
		self.y_mean = model['y_mean']
		self.y_std = model['y_std']

	def load_model(self, filename):
		with open(filename, "r") as file:
			model = json.load(file)
		return model
	
	def hypothesisFunction(self, x):
		return self.theta0 + ( self.theta1 * x )
	
	def predict(self, input):
		if not isinstance(input, (int, float)) or float(input) < 0:
			raise ValueError("Invalid input. Please enter a non-negative numeric value.")
		normalized_input = (float(input) - self.x_mean) / self.x_std
		prediction = round((self.hypothesisFunction(normalized_input) * self.y_std) + self.y_mean)
		return prediction if prediction > 0 else 0
	

if __name__ == "__main__":
	predictor = LinearRegressionPredictor()
	user_input = input("Enter the mileage of the car to predict its price: ")
	try:
		predicted_price = predictor.predict(float(user_input))
		print(f"Predicted price for a car with {user_input} km: {predicted_price}")
	except ValueError as e:
		print(e)