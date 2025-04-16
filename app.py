from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
import numpy as np
import statsmodels.api as sm

app = Flask(__name__)

X = np.array([
    [0, 19.8], [1, 23.4], [1, 27.7], [1, 24.6], [0, 21.5],
    [1, 25.1], [1, 22.4], [0, 29.3], [0, 20.8], [1, 20.2],
    [1, 27.3], [0, 24.5], [0, 22.9], [1, 18.4], [0, 24.2],
    [1, 21.0], [0, 25.9], [0, 23.2], [1, 21.6], [1, 22.8]
])
y = np.array([137, 118, 124, 124, 120, 129, 122, 142, 128, 114,
              132, 130, 130, 112, 132, 117, 134, 132, 121, 128])

model = LinearRegression().fit(X, y)

X_with_intercept = sm.add_constant(X)  # Adds the intercept term to the matrix

model_sm = sm.OLS(y, X_with_intercept).fit()

tau_hat = model_sm.params[1]  # Coefficient for the treatment indicator W (first variable in X)
tau_p_value = model_sm.pvalues[1]  # p-value for the treatment effect (W)

@app.route("/predict")
def predict():
    w = float(request.args.get("w", 0))
    x = float(request.args.get("x", 0))
    
    y_pred = model.predict([[w, x]])[0]

    with open("output.txt", "w") as f:
        f.write(f"Input: w={w}, x={x}\nPrediction: {y_pred}\n")

    return jsonify({
        "w": w,
        "x": x,
        "prediction": y_pred,
        "estimated_ATE": tau_hat,
        "p_value_of_ATE": tau_p_value
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
