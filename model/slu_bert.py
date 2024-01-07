import torch
from torch import nn



from typing import List, Tuple

import torch
from dataset.data import BIO, Label, LabelConverter
from torchcrf import CRF


class SimpleDecoder(nn.Module):
    def __init__(self, arguments,in_len: int, out_len: int):
        super().__init__()
        hidden_size = arguments.hidden_size
        self.fnn = nn.Sequential(
            nn.Linear(hidden_size, out_len),
            nn.Softmax(dim=1)
        )
        self.rnn = getattr(nn, arguments.rnn)(input_size=in_len, hidden_size=hidden_size // 2,
                                              num_layers=arguments.num_layer, batch_first=True, bidirectional=True, dropout=0.1)

    def forward(self, x):
        x = self.rnn(x)[0]
        return self.fnn(x)

class TaggingFNNCRFDecoder(nn.Module):
    def __init__(self, arguments, in_len: int, out_len: int):
        super(TaggingFNNCRFDecoder, self).__init__()
        hidden_size = arguments.hidden_size
        self.rnn = getattr(nn, arguments.rnn)(input_size=in_len, hidden_size=hidden_size // 2,
                                              num_layers=arguments.num_layer, batch_first=True, 
                                              bidirectional=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, out_len)
        self.crf = CRF(out_len, batch_first=True)

    def forward(self, x, labels=None):
        # Pass the input through the RNN layer
        x = self.rnn(x)[0]
        emissions = self.fc(x)

        if labels is not None:
            # If labels are provided, return the loss
            loss = -self.crf(emissions, labels)
            return loss
        # Otherwise, return the CRF predictions
        return self.crf.decode(emissions)



def get_output(text: List[str], output: torch.Tensor, label_converter: LabelConverter) -> List[Tuple[str, str, str]]:
    ret = []
    output = output[1:-1].argmax(dim=1)
    labels = [label_converter.index_to_label(i.item()) for i in output]
    labels.append(Label(BIO.O, '', ''))
    start = -1
    act = ''
    slot = ''
    for i, v in enumerate(labels):
        if v.bio == BIO.B:
            if start != -1:
                value = ''.join(text[start:i])
                ret.append([act, slot, value])
                start = -1
            start = i
            act = v.act
            slot = v.slot
        elif v.bio == BIO.O and start != -1:
            value = ''.join(text[start:i])
            ret.append([act, slot, value])
            start = -1
        elif v.bio == BIO.I and (v.act, v.slot) != (act, slot):
            # invalid tag sequence
            return []
    return ret
