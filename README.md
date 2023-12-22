# Automatic Speech Recognition Software based on CNN-LSTM-CTC
Automatic Speech Recognition Program based on a CNN-LSTM-CTC model.

# Package Dependencies
```
OS
Numpy 
Python 3.6
PyTorch
Torchaudio
```
# Data
The training data was obtained from subset "test-clean-100" of the LibriSpeech dataset obtained through torchaudio. The file app.py will take any waveform file as a data input.

# Commands
To install dependencies
```
pip3 install -r requirements.txt
```

To run the application
```
python app.py <path_to_waveform> <path_to_model_dict>
```

To test the model
```
python src/test.py <path_to_model_dict>
```

To train the model
```
python src/train.py
```

# Contributions
Blake worked on the app and model building / training as well as data preparation:
- app.py 
- train.py
- model.py
- various utils

Kainoa worked on model building / testing as well as data generation which involves: 
- test.py
- model.py
- various utils
