from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# ---------------------------
# Load Trained Pipeline
# ---------------------------
with open("house_price_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# Load neighborhoods once
df = pd.read_csv("train.csv")
neighborhoods = sorted(df["Neighborhood"].dropna().unique().tolist())

# ---------------------------
# Home Page
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html", neighborhoods=neighborhoods)

# ---------------------------
# Prediction Route
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        overall_qual = int(request.form["OverallQual"])
        gr_liv_area = float(request.form["GrLivArea"])
        total_bsmt_sf = float(request.form["TotalBsmtSF"])
        garage_cars = int(request.form["GarageCars"])
        year_built = int(request.form["YearBuilt"])
        neighborhood = request.form["Neighborhood"]

        input_data = pd.DataFrame([{
            "OverallQual": overall_qual,
            "GrLivArea": gr_liv_area,
            "TotalBsmtSF": total_bsmt_sf,
            "GarageCars": garage_cars,
            "YearBuilt": year_built,
            "Neighborhood": neighborhood,
        }])

        prediction = model.predict(input_data)[0]

        return render_template(
            "index.html",
            neighborhoods=neighborhoods,
            prediction_text=f"Estimated House Price: ${prediction:,.2f}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            neighborhoods=neighborhoods,
            prediction_text="Error: Please check your inputs."
        )

# ---------------------------
# Run App
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)
