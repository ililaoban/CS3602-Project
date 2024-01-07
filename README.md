# CS3602-Project

## pinyin Correction
The pinyin Correction only works when testing the dev dataset(waiting to add pinyin correction on test), not working in training.

The idea of pinyin Correction is that when the pinyin is not in the dictionary, we will find the most similar pinyin in the dictionary and use the corresponding word as the correction.

The using of pinyin correction is just add `--pinyin` when testing.
  arg_parser.add_argument('--crf', action='store_true', help='Enable CRF')

  arg_parser.add_argument('--augment', action='store_true', help='Enable data augement.')


  BERT use: HF_ENDPOINT=https://hf-mirror.com python slu_baseline_bert.py  --num_layer 1 --device 0 --hidden_size 128                                       

