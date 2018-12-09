import librosa as lbr
from tensorflow.python.keras.models import Model, load_model
import os

def prediction(model_path, output_path):
    model = load_model(model_path)
    return model

def getDefaultShape():
    tempFeatures, _ = load_track(AUDIO_DIR + "009/009152.mp3")
    return tempFeatures.shape

def load_track(filename, forceShape=None):
    sample_input, sample_rate = lbr.load(filename, mono=True)
    features = lbr.feature.melspectrogram(sample_input, **MEL_KWARGS).T
    print(features.shape)
    if forceShape is not None:
        if features.shape[0] < forceShape[0]:
            delta_shape = (forceShape[0] - features.shape[0], forceShape[1])
            features = np.append(features, np.zeros(delta_shape), axis=0)
        elif features.shape[0] > forceShape[0]:
            features = features[: forceShape[0], :]

    features[features == 0] = 1e-6

    return (np.log(features), float(sample_input.shape[0]) / sample_rate)

if __name__ == "__main__":
    model_path = os.path.join(os.path.dirname(__file__), 'Models/model_cnn50_relu.h5')
    audio_dir = os.path.join(os.path.dirname(__file__), "/Data/fma_small/")
    defaultShape = getDefaultShape(audio_dir + "009/009152.mp3")
    test_x, _ = load_track(audio_dir + "003/003270.mp3")
    print(test_x)
