def convertSentenceToFeatures(sentence):
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

    array = [c for c in sentence]
    g_1 = getNgram(array, 1)
    g_2 = getNgram(array, 2)
    g_3 = getNgram(array, 3)
    g_4 = getNgram(array, 4)
    g_5 = getNgram(array, 5)
    return [{'1g': _1, '2g': _2, '3g': _3, '4g': _4, '5g': _5} for _1, _2, _3, _4, _5 in zip(g_1, g_2, g_3, g_4, g_5)]


def convertAnnToLabels(ann_data, length):
    labels = []
    ann_ix = 0
    ann_values = ann_data.keys()
    for i in range(length):
        # end condition
        if ann_ix == len(ann_data):
            labels.append('O')
        else:
            # label
            if ann_values[ann_ix][0] == i:
                labels.append('B')
            elif ann_values[ann_ix][1] == i:
                labels.append('E')
                ann_ix += 1
            elif ann_values[ann_ix][0] < i and ann_values[ann_ix][1] > i:
                labels.append('M')
            else:
                labels.append('O')
    return labels


def convertAnnData(annotate):
    ann_data = []
    for line in annotate[:-1]:
        entity = line.split('\t')[-1]
        start = int(line.split('\t')[1].split(' ')[1])
        end = int(line.split('\t')[1].split(' ')[-1])
        ann_data.append({entity: (start, end)})
    return ann_data


def convertLabelToAnnData(sentence, labels):
    """sentenceとlabelsからannotationデータをとる。
    [{'entity':(start, end)}, ...]
    """
    ann_datas = []
    entity = ''
    start = 0
    for i, (c, label) in enumerate(zip(sentence, labels)):
        if label == 'O':
            pass
        elif label == 'B':
            start = i
            entity += c
        elif label == 'M':
            entity += c
        elif label == 'E':
            ann_datas.append({entity: (start, i)})
            entity = ''
    return ann_datas
