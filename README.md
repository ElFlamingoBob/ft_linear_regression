# ft_linear_regression

A Python GUI application that implements simple univariate linear regression from scratch using Gradient Descent. This project allows users to visualize data, train a model interactively, and make predictions.

## Features

- **Interactive GUI**: Built with `tkinter` to guide the user through dataset selection, training, and prediction.
- **Custom Implementation**: Implements the hypothesis, cost function (MSE), gradient descent algorithm, and Z-score normalization without high-level ML libraries.
- **Dataset Selection**: Choose between different datasets (e.g., `paris_housing.csv`, `car_mileage.csv`) via a dropdown menu.
- **Configurable Training**: Adjust the Learning Rate (Alpha) using a slider before training.
- **Visualization**:
  - **Data Plot**: View the scatter plot of dataset points overlaid with the trained regression line.
  - **Ex Curve**: Visualize how the prediction for a specific input value evolves over training iterations.
- **Prediction Tool**: Input a value (e.g., mileage or house size) to get a predicted price based on the trained model.
- **Parameter Inspection**: View the final learned parameters ($\theta_0, \theta_1$), RMSE, and total iterations.

## Requirements

- Python 3
- `pandas`
- `matplotlib`
- `tkinter` (usually included with Python, but may require separate installation on Linux, e.g., `sudo apt-get install python3-tk`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ElFlamingoBob/ft_linear_regression.git
   cd ft_linear_regression
   ```

2. Install the dependencies:
   ```bash
   pip install pandas matplotlib
   ```

## Usage

Run the main script to launch the application:

```bash
python src/ui.py
```

### How to use:
1. **Select File**: Choose a dataset from the dropdown list (e.g., `paris_housing.csv`).
2. **Configure**: Use the slider to set the learning rate (Alpha).
3. **Train**: Click "Start Training". The model will train for up to 10,000 iterations or until convergence.
4. **Analyze & Predict**:
   - Click **Display Plot** to see the regression line.
   - Click **Display Parameters and RMSE** to see the learned parameters and RMSE.
   - Enter a value in the text box and click **Predict Price** to test the model.

## Configuration

The application behavior is driven by `config.json`, which defines:
- Dataset filenames.
- UI labels and titles.
- Plot configuration (labels, titles).
- Specific values used for the "Ex Curve" visualization.

After training, the learned model parameters are saved to `model_weights.json` and used for predictions.
