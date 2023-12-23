import json

from utils.vocab import Vocab, LabelVocab
from utils.word2vec import Word2vecUtils
from utils.evaluator import Evaluator
from utils.pinyin import pinyin_denoise

class Example():

    @classmethod
    def configuration(cls, root, train_path=None, word2vec_path=None):
        cls.evaluator = Evaluator()
        cls.word_vocab = Vocab(padding=True, unk=True, filepath=train_path)
        cls.word2vec = Word2vecUtils(word2vec_path)
        cls.label_vocab = LabelVocab(root)

    @classmethod
    def load_dataset(cls, data_path):
        dataset = json.load(open(data_path, 'r',encoding='UTF-8'))
        examples = []
        for di, data in enumerate(dataset):
            for ui, utt in enumerate(data):
                ex = cls(utt, f'{di}-{ui}')
                examples.append(ex)
        return examples

    def __init__(self, ex: dict, did):
        super(Example, self).__init__()
        self.ex = ex
        self.did = did

        self.utt = ex['asr_1best']
        self.slot = {}
        for label in ex['semantic']:
            act_slot = f'{label[0]}-{label[1]}'
            if len(label) == 3:
                self.slot[act_slot] = label[2]
        self.tags = ['O'] * len(self.utt)
        for slot in self.slot:
            value = self.slot[slot]
            bidx = self.utt.find(value)
            if bidx != -1:
                self.tags[bidx: bidx + len(value)] = [f'I-{slot}'] * len(value)
                self.tags[bidx] = f'B-{slot}'
        self.slotvalue = [f'{slot}-{value}' for slot, value in self.slot.items()]
        self.input_idx = [Example.word_vocab[c] for c in self.utt]
        l = Example.label_vocab
        self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]

    @classmethod
    def pinyin_correction(cls, predictions):
        select_pos_set = {'poi名称', 'poi修饰', 'poi目标', '起点名称', '起点修饰', '起点目标', '终点名称', '终点修饰', '终点目标', '途经点名称'}
        select_others_set = {'请求类型': [cls.label_vocab.request_map_dic, cls.label_vocab.request_pinyin_set], \
        '出行方式' : [cls.label_vocab.travel_map_dic, cls.label_vocab.travel_pinyin_set], \
        '路线偏好' : [cls.label_vocab.route_map_dic, cls.label_vocab.route_pinyin_set], \
        '对象' :  [cls.label_vocab.object_map_dic, cls.label_vocab.object_pinyin_set], \
        '页码' : [cls.label_vocab.page_map_dic, cls.label_vocab.page_pinyin_set], \
        '操作' : [cls.label_vocab.opera_map_dic, cls.label_vocab.opera_pinyin_set], \
        '序列号' : [cls.label_vocab.ordinal_map_dic, cls.label_vocab.ordinal_pinyin_set]   }
        return pinyin_denoise(
                            predictions = predictions, 
                            select_pos_set = select_pos_set, 
                            select_others_set = select_others_set,
                            example_map_dic=cls.label_vocab.poi_map_dic,
                            example_pinyin_set=cls.label_vocab.poi_pinyin_set)