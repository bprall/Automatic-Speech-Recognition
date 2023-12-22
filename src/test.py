import os
import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torchaudio

from utility import *
from model import *

BATCH_SIZE = 10

test_set = "test-clean"

def test(model, device, test_loader, criterion, epoch, iter_meter):
    print('\nevaluating...')
    model.eval()
    test_loss = 0
    test_cer = []
    test_wer = []

    with torch.no_grad():
        for i, _data in enumerate(test_loader):
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            output = model(spectrograms)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)

            loss = criterion(output, labels, input_lengths, label_lengths)
            test_loss += loss.item() / len(test_loader)

            decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths)

            print("Decoded_pred len:", len(decoded_preds))
            print("Decoded_targets len:", len(decoded_targets))

            for j in range(len(decoded_preds)):
                    test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                    test_wer.append(wer(decoded_targets[j], decoded_preds[j]))

    avg_cer = sum(test_cer) / len(test_cer)
    avg_wer = sum(test_wer) / len(test_wer)

    print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss, avg_cer, avg_wer))


def main(model_dict_path, batch_size=BATCH_SIZE, test_url="test-clean"):
    hparams = {
        "n_cnn_layers": 3,
        "n_lstm_layers": 5,
        "lstm_dim": 256,
        "n_class": 29,
        "n_feats": 128,
        "dropout": 0.1,
        "batch_size": batch_size,
    }

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")

    if not os.path.isdir("./data"):
        os.makedirs("./data")

    test_dataset = torchaudio.datasets.LIBRISPEECH("./data", url=test_url, download=True)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=hparams['batch_size'],
                                  shuffle=False, collate_fn=lambda x: data_processing(x, 'valid'), **kwargs)

    model = SpeechRecognitionModel(
        hparams['n_cnn_layers'], hparams['n_lstm_layers'], hparams['lstm_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['dropout']
        )

    state_dict = torch.load(model_dict_path)
    model.load_state_dict(state_dict)
    model = model.to(device)

    print(model)
    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    criterion = nn.CTCLoss(blank=28).to(device)

    iter_meter = IterMeter()
    test(model, device, test_loader, criterion, epoch, iter_meter)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python test.py <path_to_model_dict>")
        sys.exit(1)

    model_dict_path = sys.argv[1]
    
    main(model_dict_path, BATCH_SIZE, test_set)
