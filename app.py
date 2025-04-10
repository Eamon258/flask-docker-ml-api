from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

# Training data
X = np.array([[0], [1], [1], [1], [0], 
              [1], [1], [0], [O], [1], 
              [1], [0], [0], [1], [0],
              [1], [0], [0], [1], [1]],
             [[19.8], [23.4], [27.7], [24.6], [21.5],
              [25.1], [22.4], [29.3], [20.8], [20.2],
              [27.3], [24.5], [22.9], [18.4], [24.2],
              [21], [25.9], [23.2], [21.6], [22.8]])
y = np.array([137, 118, 124, 124, 120, 129, 122, 142, 
              128, 114, 132, 130, 130, 112, 132, 117, 
              134, 132, 121, 128])
model = LinearRegression().fit(X, y)

@app.route("/predict")
def predict():
    x = float(request.args.get("x", 0))
    y_pred = model.predict([[x]])[0]
    
    # Log prediction
    with open("output.txt", "w") as f:
        f.write(f"Input x: {x}\nPrediction: {y_pred}\n")
    
    return jsonify({"x": x, "prediction": y_pred})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
