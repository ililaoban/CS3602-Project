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

    def __init__(self, arguments, input_size, num_tags):
        super(TaggingFNNCRFDecoder, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def loss_func(self, logits, labels, mask):
        # print(self.crf.forward(logits, labels, mask, reduction='mean'))

        return -self.crf.forward(logits, labels, mask, reduction='mean')
        
        
        
    def forward(self, hiddens, mask, labels=None):
        logits = self.output_layer(hiddens)
        pred = self.crf.decode(logits, mask)
        # print(len(pred), len(pred[4])) # bsize x seqlen 列表里的长度和句子本身的长度是一一对应的
        #pred = torch.tensor(pred)
        
        if labels is not None:
            loss = self.loss_func(logits, labels, mask)
            return pred, loss
        return pred, None



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
