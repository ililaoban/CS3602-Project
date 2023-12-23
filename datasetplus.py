import json
import sys
import os, time, gc
import copy
# import synonyms
# 手动构建同类词词典（此处近义词为替换后语义通顺即可，比如“开始导航”可以替换为“重新导航”
DaoHang = ["导航", "开始导航", "继续导航", "保持导航", "恢复导航", "导"]
def generate_ontology_file(data_path):
    ontology_file = json.load(open(os.path.join(data_path, "ontology.json"), 'r', encoding='utf-8'))
    slots_dict = ontology_file['slots']
    final_output = []
    for key, value in slots_dict.items():
        if isinstance(value, str):
            txt_file = open(os.path.join(data_path, value), 'r', encoding='utf-8')
            words = txt_file.read().splitlines()
            if key == 'poi名称' or key=='操作' or key=='序列号':
                for word in words:
                    tmp_output = []
                    data_dict = {}
                    data_dict['utt_id'] = 1
                    data_dict['manual_transcript'] = word
                    data_dict['asr_1best'] = word
                    data_dict['semantic'] = [["inform", key, word]]
                    tmp_output.append(data_dict)
                    final_output.append(tmp_output)
            elif key == '终点名称':
                for word in words:
                    tmp_output = []
                    data_dict = {}
                    data_dict['utt_id'] = 1
                    data_dict['manual_transcript'] = "导航到" + word
                    data_dict['asr_1best'] = "导航到" + word
                    data_dict['semantic'] = [["inform", key, word], ["inform", '操作', '导航']]
                    tmp_output.append(data_dict)
                    final_output.append(tmp_output)
            
        else:
            if key != '对象':
                for sub_value in value:
                    tmp_output = []
                    data_dict = {}
                    data_dict['utt_id'] = 1
                    data_dict['manual_transcript'] = sub_value
                    data_dict['asr_1best'] = sub_value
                    data_dict['semantic'] = [["inform", key, sub_value]]
                    tmp_output.append(data_dict)
                    final_output.append(tmp_output)
    return final_output





def main():
    data_path = "./data"
    train_path = os.path.join(data_path, 'train.json')
    train_dataset = json.load(open(train_path, 'r', encoding='utf-8'))
    ontology_file = generate_ontology_file(data_path)
    train_dataset += ontology_file
    json_str=json.dumps(train_dataset,indent=4,ensure_ascii=False)
    with open(os.path.join(data_path, 'augmented_train_with_ontology.json'),'w',encoding='utf-8') as f:
        f.write(json_str)
if __name__=='__main__':
    main()