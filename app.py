import os
import sys
import torch
import torch.nn.functional as F
import torchaudio

from .utils.processing import *
from .utils.itermeter import *
from .model.model import *
from .transform import *

def GreedyDecoder(output, blank_label=28, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    for i, args in enumerate(arg_maxes):
        decode = []
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j - 1]:
                    continue
                decode.append(index.item())
        decodes.append(text_transform.int_to_text(decode))
    return decodes

def transcribe(model, device, spectrogram):    
    model.eval()
    spectrogram = spectrogram.to(device)

    with torch.no_grad():
        output = model(spectrogram)
    output = F.log_softmax(output, dim=2)
    output = output.transpose(0, 1)

    decoded_pred = GreedyDecoder(output.transpose(0, 1))

    print(decoded_pred)

def main(waveform_path, model_dict_path):
    hparams = {
        "n_cnn_layers": 3,
        "n_lstm_layers": 5,
        "lstm_dim": 256,
        "n_class": 29,
        "n_feats": 128,
        "dropout": 0.1,
    }

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")

    if not os.path.isdir("./data"):
        os.makedirs("./data")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    waveform, _ = torchaudio.load(waveform_path)

    spectrogram = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
    spectrogram = spectrogram.unsqueeze(0).unsqueeze(1).transpose(2, 3)

    model = SpeechRecognitionModel(
        hparams['n_cnn_layers'], hparams['n_lstm_layers'], hparams['lstm_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['dropout']
    ).to(device)

    state_dict = torch.load(model_dict_path)
    model.load_state_dict(state_dict)
    model = model.to(device)

    transcribe(model, device, spectrogram)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python app.py <path_to_waveform> <path_to_model_dict>")
        sys.exit(1)

    waveform_path = sys.argv[1]
    model_dict_path = sys.argv[2]
    main(waveform_path, model_dict_path)
