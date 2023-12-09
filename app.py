import os
import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
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


def transcribe(model, device, test_loader, iter_meter):
    decodes = []
    
    model.eval()
    
    for i, _data in enumerate(test_loader):
        spectrograms, labels, input_lengths, label_lengths = _data
        spectrograms = spectrograms.to(device)

        with torch.no_grad():
            output = model(spectrograms)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)

        decoded_preds = GreedyDecoder(output.transpose(0, 1))

        for j in range(len(decoded_preds)):
            decodes.append(decoded_pred[j])
            
    for i in range(len(decodes)):
        print(decodes[j])


def main(batch_size=BATCH_SIZE, dataset):
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
    
    test_loader = data.DataLoader(dataset=dataset, batch_size=hparams['batch_size'],
                                  shuffle=False, collate_fn=lambda x: data_processing(x, 'valid'), **kwargs)

    trained_model = SpeechRecognitionModel(
        hparams['n_cnn_layers'], hparams['n_lstm_layers'], hparams['lstm_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['dropout']
        ).to(device)

    state_dict = torch.load("final_speech_to_text_model.pth")

    trained_model.load_state_dict(state_dict)

    trained_model = trained_model.to(device)

    print(model)
    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    criterion = nn.CTCLoss(blank=28).to(device)

    iter_meter = IterMeter()
    transcribe(model, device, test_loader, iter_meter)
    

if __name__ == '__main__':
    main(BATCH_SIZE, test_set)
