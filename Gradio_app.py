import gradio as gr
from joblib import load

# Define the features
features = load('features_list.pkl') 
# Function to load models
def load_model(model_name):
    return load(f'{model_name}.pkl')

# Prediction function
def predict(*inputs):
    input_data = [[float(i) for i in inputs]]
    
    logistic_regression_model = load_model('logistic_regression_model')
    svm_model = load_model('svm_model')
    random_forest_model = load_model('random_forest_model')
    
    lr_prediction = logistic_regression_model.predict(input_data)[0]
    svm_prediction = svm_model.predict(input_data)[0]
    rf_prediction = random_forest_model.predict(input_data)[0]
    
    results = f"Logistic Regression: {'Malignant' if lr_prediction == 1 else 'Benign'}, SVM: {'Malignant' if svm_prediction == 1 else 'Benign'}, Random Forest: {'Malignant' if rf_prediction == 1 else 'Benign'}"
    
    return results

# Define the Gradio interface
inputs = [gr.Number(label=feature) for feature in features]
output = gr.Text()

interface = gr.Interface(fn=predict, inputs=inputs, outputs=output, title="Breast Cancer Prediction")
interface.launch(share=True) 
