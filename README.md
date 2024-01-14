# CS3602-Project

## pinyin Correction(bo huang)

This branch implements the pinyin baseline.

The pinyin Correction only works when testing the dev dataset(waiting to add pinyin correction on test), not working in training.

The idea of pinyin Correction is that when the pinyin is not in the dictionary, we will find the most similar pinyin in the dictionary and use the corresponding word as the correction.

The using of pinyin correction is just add `--pinyin_correction` when testing.

The command to run
```
python scripts/slu_baseline.py --pinyin_correction
```