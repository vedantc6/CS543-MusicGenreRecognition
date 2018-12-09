from tensorflow.python.keras.models import Model, load_model
import tensorflowjs as tfjs
import os

def extract_realtime_model(full_model):
    input = full_model.get_layer('input').input
    output = full_model.get_layer('output_realtime').output
    model = Model(inputs=input, outputs=output)
    return model

def convert_to_js(model_path, output_path):
    model = load_model(model_path)
    realtime_model = extract_realtime_model(model)
    realtime_model.compile(optimizer=model.optimizer, loss=model.loss)
    tfjs.converters.save_keras_model(realtime_model, output_path)

if __name__ == "__main__":
    model_path = os.path.join(os.path.dirname(__file__), 'Models/model_cnn50_relu.h5')
    output_path = os.path.join(os.path.dirname(__file__), 'static/model')
    convert_to_js(model_path, output_path)
