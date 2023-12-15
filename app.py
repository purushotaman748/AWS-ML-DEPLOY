from flask import Flask, render_template, request
import pickle

# Load the trained model
with open("car_price_model.pkl", "rb") as f:
  model = pickle.load(f)

app = Flask(__name__)

def format_price(value, decimals=2, currency="$"):
  """
  Formats a number with the specified number of decimal places and currency symbol.
  """
  return f"{currency}{value:.{decimals}f}"

@app.route("/")
def index():
  return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
  if request.method == "POST":
    distance = float(request.form["distance"])
    age = float(request.form["age"])
    
    # Prepare features as a list
    features = [distance, age]
    
    # Predict price based on features
    predicted_price = model.predict([features])[0]
    
    return render_template("index.html", predicted_price=predicted_price)
  else:
    return "Invalid request method"

if __name__ == '__main__':

    app.run(host="0.0.0.0", port=8080) #for deployment run
    #app.run(host="127.0.0.1", port=8000,debug=True) # for local run
