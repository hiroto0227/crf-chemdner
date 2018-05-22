def getNgram(array, n):
    """targetとなる語は一番後ろ。
    [w_(i-2), w_(i-1), 2_i]
    """
    ret_array = []
    pagging_array = ['<B>'] * (n - 1) + array + ['<E>'] * (n - 1)
    for i in range(len(pagging_array)):
        ret_array.append(''.join(pagging_array[i:i + n]))
    if n == 1:
        return ret_array
    else:
        return ret_array[:-(n - 1)]


class Transformer:

    def __init__(self):
        pass

    def convertTextToNgram(self, text):
        """textをn-gramの辞書にして返す。
        """
        array = [c for c in text]
        g_1 = getNgram(array, 1)
        g_2 = getNgram(array, 2)
        g_3 = getNgram(array, 3)
        g_4 = getNgram(array, 4)
        g_5 = getNgram(array, 5)
        return [{'1g': _1, '2g': _2, '3g': _3, '4g': _4, '5g': _5}
                for _1, _2, _3, _4, _5 in zip(g_1, g_2, g_3, g_4, g_5)]

    def convertAnnsToLabels(self, anns, length):
        """annsを入力し、labelsで出力
        length : labelsの長さを返す。
        """
        labels = []
        ann_ix = 0
        ann_values = [v for ann in anns for k, v in ann.items()]
        for i in range(length):
            # end condition
            if ann_ix == len(anns):
                labels.append('O')
            else:
                # label
                if ann_values[ann_ix][0] == ann_values[ann_ix][1]:
                    labels.append('S')
                elif ann_values[ann_ix][0] == i:
                    labels.append('B')
                elif ann_values[ann_ix][1] == i:
                    labels.append('E')
                    ann_ix += 1
                elif ann_values[ann_ix][0] < i and ann_values[ann_ix][1] > i:
                    labels.append('M')
                else:
                    labels.append('O')
        return labels

    def convertLabelsToAnn(self, text, labels):
        """sentenceとlabelsからannotationデータをとる。
        [{'entity':(start, end)}, ...]
        """
        ann_datas = []
        entity = ''
        start = 0
        for i, (c, label) in enumerate(zip(text, labels)):
            if label == 'O':
                pass
            elif label == 'S':
                ann_datas.append({c: (i, i)})
            elif label == 'B':
                start = i
                entity += c
            elif label == 'M':
                entity += c
            elif label == 'E':
                ann_datas.append({entity: (start, i)})
                entity = ''
        return ann_datas
