# CS3602-Project

## pinyin Correction
The pinyin Correction only works when testing the dev dataset(waiting to add pinyin correction on test), not working in training.

The idea of pinyin Correction is that when the pinyin is not in the dictionary, we will find the most similar pinyin in the dictionary and use the corresponding word as the correction.

The using of pinyin correction is just add `--pinyin` when testing.
  arg_parser.add_argument('--crf', action='store_true', help='Enable CRF')

  arg_parser.add_argument('--augment', action='store_true', help='Enable data augement.')

