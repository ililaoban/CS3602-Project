from xpinyin import Pinyin


# calculate the similarity of two pinyin
def get_pinyin_similarity(standard_pinyin, tmp_pinyin) :
    standard_set = set (standard_pinyin.split(' '))
    tmp_set = set (tmp_pinyin.split(' '))

    inter_set = standard_set & tmp_set
    similarity = len (inter_set) / (len (standard_set) + len (tmp_set) )
    return similarity


def get_standard_output(map_dic, pinyin_set, tmp_pinyin) :
    if tmp_pinyin in pinyin_set :
        standard_output = map_dic[tmp_pinyin]
    else :
        max_similarity = 0
        most_similar_pinyin = ''
        for standard_pinyin in pinyin_set :
            similarity = get_pinyin_similarity(standard_pinyin, tmp_pinyin)
            if similarity > max_similarity :
                max_similarity = similarity
                most_similar_pinyin = standard_pinyin
        if max_similarity == 0 : 
            standard_output = '无'
        else :
            standard_output = map_dic[most_similar_pinyin]
    return standard_output

def pinyin_denoise(predictions, select_pos_set, select_others_set, example_map_dic, example_pinyin_set):
    p = Pinyin()
    modify_num = 0
    for i, pred in enumerate(predictions):
        pred_length = len(pred)
        if pred_length > 0 :
            for j in range(pred_length):
                tmp_pred = pred[j]
                split_result = tmp_pred.split('-')
                tmp_pinyin = p.get_pinyin(split_result[2], ' ')
                if split_result[1] != 'value' :
                    if split_result[1] in select_pos_set:
                        map_dic, pinyin_set = example_map_dic, example_pinyin_set
                    else :
                        [map_dic, pinyin_set] = select_others_set[split_result[1]]

                    standard_output = get_standard_output(map_dic, pinyin_set, tmp_pinyin)
                    modify_pred = split_result[0] + '-' + split_result[1] + '-' + standard_output
                    if standard_output != split_result[2]:
                        modify_num += 1
                    predictions[i][j] = modify_pred
    return predictions



# def anti_noise_prediction(predictions, Example):
#     p = Pinyin()
#     select_pos_set = {'poi名称', 'poi修饰', 'poi目标', '起点名称', '起点修饰', '起点目标', '终点名称', '终点修饰', '终点目标', '途经点名称'}
#     select_others_set = {'请求类型': [Example.label_vocab.request_map_dic, Example.label_vocab.request_pinyin_set], \
#         '出行方式' : [Example.label_vocab.travel_map_dic, Example.label_vocab.travel_pinyin_set], \
#         '路线偏好' : [Example.label_vocab.route_map_dic, Example.label_vocab.route_pinyin_set], \
#         '对象' :  [Example.label_vocab.object_map_dic, Example.label_vocab.object_pinyin_set], \
#         '页码' : [Example.label_vocab.page_map_dic, Example.label_vocab.page_pinyin_set], \
#         '操作' : [Example.label_vocab.opera_map_dic, Example.label_vocab.opera_pinyin_set], \
#         '序列号' : [Example.label_vocab.ordinal_map_dic, Example.label_vocab.ordinal_pinyin_set]   }

#     modify_num = 0
#     for i, pred in enumerate(predictions):
#         pred_length = len(pred)
#         if pred_length > 0 :
#             for j in range(pred_length):
#                 tmp_pred = pred[j]
#                 split_result = tmp_pred.split('-')
#                 tmp_pinyin = p.get_pinyin(split_result[2], ' ')
#                 if split_result[1] != 'value' :
#                     if split_result[1] in select_pos_set :
#                         map_dic, pinyin_set = Example.label_vocab.poi_map_dic, Example.label_vocab.poi_pinyin_set
#                     else :
#                         [map_dic, pinyin_set] = select_others_set[split_result[1]]

#                     standard_output = get_standard_output(map_dic, pinyin_set, tmp_pinyin)
#                     modify_pred = split_result[0] + '-' + split_result[1] + '-' + standard_output
#                     if standard_output != split_result[2] :
#                         modify_num += 1
#                     # Bob thinks comment the folliwing line, is turning off the function of anti_noise_prediction
#                     predictions[i][j] = modify_pred
#     print ("modify_num == ", modify_num)                    
#     return  predictions
