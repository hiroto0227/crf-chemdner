from chemdataextractor.nlp.tokenize import ChemWordTokenizer

def getNgram(array, n_range=(-2, 2), mode='letter'):
    """n_range[0]からn_range[1]までのiの周辺語を繋げて返す。
    >>> getNgram(['I', ' ', 'a', 'm', ' ', 'a', ' ', 'm', 'a', 'n', '.'], (-2, 2))
    >>> ['<B><B>I a', '<B>I am', 'I am ', ' am a', 'am a ' ...]
    """
    assert n_range[0] <= n_range[1], 'invalid n_range. (n_range[0] <= n_range[1])'
    ret_array = []
    pagging_array = ['<B>'] * max(0, -1 * n_range[0]) + array + ['<E>'] * max(n_range[1], 0)
    for i in range(len(array)):
        if n_range[0] > 0:
            start_ix = i + n_range[0]
        else:
            start_ix = i
        end_ix = start_ix + (n_range[1] - n_range[0]) + 1
        if mode == 'letter':
            ret_array.append(''.join(pagging_array[start_ix: end_ix]))
        elif mode == 'word':
            ret_array.append(' '.join(pagging_array[start_ix: end_ix]))
    return ret_array


class LetterLevelTransformer:
    def __init__(self):
        pass

    def convertTextToFeatures(self, text):
        """textをn-gramの辞書にして返す。
        """
        array = [c for c in text]
        g_1_1 = getNgram(array, (-4, -4), mode='letter')
        g_1_2 = getNgram(array, (-3, -3), mode='letter')
        g_1_3 = getNgram(array, (-2, -2), mode='letter')
        g_1_4 = getNgram(array, (-1, -1), mode='letter')
        g_1_5 = getNgram(array, (0, 0), mode='letter')
        g_1_6 = getNgram(array, (1, 1), mode='letter')
        g_1_7 = getNgram(array, (2, 2), mode='letter')
        g_1_8 = getNgram(array, (3, 3), mode='letter')
        g_1_9 = getNgram(array, (4, 4), mode='letter')
        g_2_1 = getNgram(array, (-1, 0), mode='letter')
        g_2_2 = getNgram(array, (0, 1), mode='letter')
        g_3_1 = getNgram(array, (-2, 0), mode='letter')
        g_3_2 = getNgram(array, (-1, 1), mode='letter')
        g_3_3 = getNgram(array, (0, 2), mode='letter')
        g_4_1 = getNgram(array, (-3, 0), mode='letter')
        g_4_2 = getNgram(array, (-2, 1), mode='letter')
        g_4_3 = getNgram(array, (-1, 2), mode='letter')
        g_4_4 = getNgram(array, (0, 3), mode='letter')
        g_5_1 = getNgram(array, (-4, 0), mode='letter')
        g_5_2 = getNgram(array, (-3, 1), mode='letter')
        g_5_3 = getNgram(array, (-2, 2), mode='letter')
        g_5_4 = getNgram(array, (-1, 3), mode='letter')
        g_5_5 = getNgram(array, (0, 4), mode='letter')
        return [{'a': _1_1, 'a_2': _1_2, 'a_3': _1_3, 'a_4': _1_4, 'a_5': _1_5, 'a_6': _1_6, 'a_7': _1_7, 'a_8': _1_8, 'a_9': _1_9, 'b': _2_1, 'c': _2_2, 'd': _3_1, 'e': _3_2, 'f': _3_3, 'g': _4_1, 'h': _4_2, 'i': _4_3, 'j': _4_4, 'k': _5_1, 'l': _5_2, 'm': _5_3, 'n': _5_4, 'o': _5_5} for _1_1, _1_2, _1_3, _1_4, _1_5, _1_6, _1_7, _1_8, _1_9, _2_1, _2_2, _3_1, _3_2, _3_3, _4_1, _4_2, _4_3, _4_4, _5_1, _5_2, _5_3, _5_4, _5_5 in zip(g_1_1, g_1_2, g_1_3, g_1_4, g_1_5, g_1_6, g_1_7, g_1_8, g_1_9, g_2_1, g_2_2, g_3_1, g_3_2, g_3_3, g_4_1, g_4_2, g_4_3, g_4_4, g_5_1, g_5_2, g_5_3, g_5_4, g_5_5)]

    def convertAnnsToLabels(self, anns, text):
        """annsを入力し、labelsで出力
        length : labelsの長さを返す。
        """
        labels = []
        ann_ix = 0
        ann_values = [v for ann in anns for k, v in ann.items()]
        for i in range(len(text)):
            # end condition
            if ann_ix == len(anns):
                labels.append('O')
            else:
                # label
                if ann_values[ann_ix][0] == ann_values[ann_ix][1]:
                    labels.append('S')
                    ann_ix += 1
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
                entity += c
                ann_datas.append({entity: (start, i)})
                entity = ''
        return ann_datas


class WordLevelTransformer():
    def __init__(self):
        self.cwt = ChemWordTokenizer()

    def convertTextToFeatures(self, text):
        """textをn-gramの辞書にして返す。
        """
        array = self.cwt.tokenize(text)
        g_1_1 = getNgram(array, (-2, -2), mode='word')
        g_1_2 = getNgram(array, (-1, -1), mode='word')
        g_1_3 = getNgram(array, (0, 0), mode='word')
        g_1_4 = getNgram(array, (1, 1), mode='word')
        g_1_5 = getNgram(array, (2, 2), mode='letter')
        g_2_1 = getNgram(array, (-2, -1), mode='letter')
        g_2_2 = getNgram(array, (-1, 0), mode='letter')
        g_2_3 = getNgram(array, (0, 1), mode='letter')
        g_2_4 = getNgram(array, (1, 2), mode='letter')
        g_3_1 = getNgram(array, (-3, 0), mode='letter')
        g_3_2 = getNgram(array, (-2, 1), mode='letter')
        g_3_3 = getNgram(array, (-1, 2), mode='letter')
        g_3_4 = getNgram(array, (0, 3), mode='letter')
        return [{'1_1': _1_1, '1_2': _1_2, '1_3': _1_3, '1_4': _1_4, '1_5': _1_5, '2_1': _2_1, '2_2': _2_2, '2_3': _2_3, '2_4': _2_4, '3_1': _3_1, '3_2': _3_2, '3_3': _3_3, '3_4': _3_4} for _1_1, _1_2, _1_3, _1_4, _1_5, _2_1, _2_2, _2_3, _2_4, _3_1, _3_2, _3_3, _3_4 in zip(g_1_1, g_1_2, g_1_3, g_1_4, g_1_5, g_2_1, g_2_2, g_2_3, g_2_4, g_3_1, g_3_2, g_3_3, g_3_4)]
    
    def convertAnnsToLabels(self, anns, text):
        """annsを入力し、labelsで出力, word lebel
        >>> convertAnnsToWordLabels(anns)
        >>> ['O', 'O', 'B', 'E']
        """
        ann_values = [v for ann in anns for k, v in ann.items()]
        labels = []
        ann_ix = 0
        #print('★' * 30)
        #print(anns)
        for token, span in zip(self.cwt.tokenize(text), self.cwt.span_tokenize(text)):
            if ann_ix == len(ann_values):
                labels.append('O')
            else:
                #print('----')
                #print(token, span)
                # お互いのstartが同じだった場合。
                if span[0] == ann_values[ann_ix][0] and span[1] == ann_values[ann_ix][1]:
                    labels.append('S')
                    ann_ix += 1
                elif span[0] == ann_values[ann_ix][0] and span[1] < ann_values[ann_ix][1]:
                    labels.append('B')
                elif span[1] == ann_values[ann_ix][1] and span[0] > ann_values[ann_ix][0]:
                    labels.append('E')
                    ann_ix += 1
                elif ann_values[ann_ix][0] < span[0] and ann_values[ann_ix][1] > span[1]:
                    labels.append('M')
                else:
                    labels.append('O')
        #print(labels)
        return labels

    def convertLabelsToAnn(self, text, labels):
        """sentenceとlabelsからannotationデータをとる。
        [{'entity':(start, end)}, ...]
        """
        ann_datas = []
        entity = ''
        start = 0
        for i, (token, span, label) in enumerate(zip(self.cwt.tokenize(text), self.cwt.span_tokenize(text), labels)):
            if label == 'O':
                pass
            elif label == 'S':
                ann_datas.append({token: (span[0], span[1])})
            elif label == 'B':
                start = span[0]
                entity += token
            elif label == 'M':
                entity += ' ' + token
            elif label == 'E':
                entity += ' ' + token
                ann_datas.append({entity: (start, span[1])})
                entity = ''
        return ann_datas
