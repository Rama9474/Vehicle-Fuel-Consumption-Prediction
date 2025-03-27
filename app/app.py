from flask import Flask, render_template, request
import pickle


# Load the trained model (Ensure the path to 'fuel2.pkl' is correct)
model = pickle.load(open(r'C:\Users\komma\Documents\fuel-consumption-prediction\app\model\fuel2.pkl', 'rb'))

# # Initialize Flask app
# app = Flask(__name__)
# Absolute path to the 'templates' directory
template_dir = r'C:\Users\komma\Documents\fuel-consumption-prediction\app\templates'

# Initialize Flask app with custom template directory
app = Flask(__name__, template_folder=template_dir)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for the about page
@app.route('/About')
def about():
    return render_template('About.html')

# Route for the contact page
@app.route('/Contact')
def contact():
    return render_template('Contact.html')

# Route for making a prediction
@app.route('/y_predict', methods=['POST', 'GET'])
def y_predict():
    return render_template('y_predict.html')

# Route for displaying prediction result
@app.route('/Result', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Extract input features from the form
        x_test = [[float(x) for x in request.form.values()]]
        print('Actual Input:', x_test)

        # Make prediction using the loaded model
        pred = model.predict(x_test)

        # Display result on the result page
        return render_template('result.html', prediction_text=f'Car fuel consumption (L/100km): {pred[0]:.2f}')
    
    # In case of GET request, just render the result page without prediction
    return render_template('Result.html')

if __name__ == '__main__':
    app.run(debug=True)
