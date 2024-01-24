from flask import Flask,render_template,request
import pickle
import sklearn.metrics
import sklearn

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/prediction', methods = ['GET','POST'])
def predict():

    
    if request.method == 'POST':
       
        # Getting user input from the form
        feature_values = [float(request.form['sepal_length']),float(request.form['sepal_width']),float(request.form['petal_length']),float(request.form['petal_width'])]
        print(feature_values)
        # Load the pickled model
        try:
            with open('iris_svm_model.pkl', 'rb') as model_file:
                loaded_model = pickle.load(model_file)
            print("Model loaded successfully")
            
            # Perform additional checks or verification steps here if needed

        except Exception as e:
            print(f"Error loading model: {str(e)}")
        # model = pickle.load(open('iris_model.pkl', 'rb'))
            
        prediction = loaded_model.predict([feature_values])[0]
        species_mapping = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
        predicted_species = species_mapping[prediction]

    return render_template('result.html', species=predicted_species) 



if __name__ == '__main__':
    app.run(debug=True)