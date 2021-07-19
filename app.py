from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

# Defining Flask app
app = Flask(__name__)

# Loading the trained ML models
model = pickle.load(open('./models/model.pkl', 'rb'))

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Lifestyle
@app.route('/lifestyle')
def lifestyle():
    return render_template('lifestyle.html')

# Healthy Food
@app.route('/food')
def food():
    return render_template('food.html')

# Yoga
@app.route('/yoga')
def yoga():
    return render_template('yoga.html')

# Meditation
@app.route('/meditation')
def meditation():
    return render_template('meditation.html')

# Wim Hof Breathing
@app.route('/breathing')
def breathing():
    return render_template('breathing.html')

# BMI
@app.route('/bodymass', methods=['GET', 'POST'])
def bodymass():
    result = None
    bmi = None
    fat = None
    if request.method == 'POST':
        w = request.form["Weight"]
        h = request.form["Height"]
        a = request.form["Age"]
        g = request.form["Gender"]
        h = float(h) * 0.3048
        bmi = round((float(w) / (h) ** 2), 1)
        m_fat = round((1.20 * float(bmi)) + (0.23 * float(a)) - 16.2, 1)
        f_fat = round((1.20 * float(bmi)) + (0.23 * float(a)) - 5.4, 1)
        if bmi < 18.5:
            result = f'"underweight"'
        elif bmi >= 18.5 and bmi < 24.9:
            result = 'in "healthy weight" category'
        else:
            result = "overweight"
        if g == "Male":
            fat = m_fat
        else:
            fat = f_fat
    return render_template('bmi.html', result=f'You are {result} with the BMI score: {bmi} kg/m2. \n Your body fat percentage is {fat}%.')


# Breast cancer
@app.route("/cancer", methods=['GET', 'POST'])
def cancer():
    output = None
    if request.method == 'POST':
        # Converting the multiple inputs into numpy array
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        # Tumor columns in dataset
        features_name = ['clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
                        'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
                        'bland_chromatin', 'normal_nucleoli', 'mitoses']

        # Predicting the input values
        df = pd.DataFrame(final_features, columns=features_name)
        output = model.predict(df)
        
        if output == 4:
            output = "Malignant tumor: You have cancer."
        else:
            output = "Benign tumor: You don't have cancer."

    return render_template('cancer.html', prediction=f'{output}')


# Run the app
if __name__ == "__main__":
    app.run(debug=True)