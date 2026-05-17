import tgt
import chardet
from pprint import pprint

from collections import defaultdict
from utils import SAMPLES_TESTING
import pandas as pd

def confusion_to_df(confusion):
    df = pd.DataFrame([
        {"ref": k[0], "hyp": k[1], "time": v}
        for k, v in confusion.items()
    ])
    return df

def to_matrix(df):
    matrix = df.pivot_table(
        index="ref",
        columns="hyp",
        values="time",
        aggfunc="sum",
        fill_value=0
    )
    return matrix

import matplotlib.pyplot as plt

def plot_confusion(matrix, iso):
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, aspect="auto")

    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=90)
    plt.yticks(range(len(matrix.index)), matrix.index)

    plt.colorbar(label="total time (s)")
    plt.title(f"Confusion Matrix {iso.upper()} (Golden label X/Misslabeled as Y)")
    plt.tight_layout()
    plt.show()

def remove_diagonal(matrix):
    m = matrix.copy()
    for i in m.index:
        if i in m.columns:
            m.loc[i, i] = 0
    return m



def load_intervals(filepath):
    intervals = []

    with open(filepath, 'rb') as f:
        encoding = chardet.detect(f.read())['encoding']
    tg = tgt.read_textgrid(filepath,encoding=encoding)
    tier = tg.get_tier_by_name('phones')  

    for interval in tier:
        intervals.append({
            'start':round(interval.start_time,3),
            'end':round(interval.end_time,3),
            'text':interval.text})
    return intervals

def sentence_overlap(seq1, seq2):
    # returns time where symbols match between seq1 and seq2
    time_correct = 0.0
    time_total = 0.0
    index1=index2=0

    while index1 < len(seq1) and index2 < len(seq2):
        el1 = seq1[index1]
        el2 = seq2[index2]
        # Find overlap
        overlap_start = max(el1['start'], el2['start'])
        overlap_end = min(el1['end'], el2['end'])

        if overlap_start < overlap_end:
            overlap_duration = overlap_end - overlap_start
            time_total += overlap_duration
            if el1['text']== el2['text']:
                time_correct += overlap_duration

        # Advance whichever interval ends first
        if el1['end']<= el2['end']:
            time_total += el1['end']- el1['start']
            index1 += 1
        else:
            time_total += el2['end']- el2['start']
            index2 += 1

    return time_correct, time_total


def confusion_matrix(seq1, seq2):
    """Returns confusion matrix as dict of (sym1, sym2) -> time"""

    confusion = defaultdict(float)   # default 0.0
    index1 = index2 = 0
    print(seq1)
    while index1 < len(seq1) and index2 < len(seq2):
        el1 = seq1[index1]
        el2 = seq2[index2]
        overlap_start = max(el1['start'], el2['start'])
        overlap_end = min(el1['end'], el2['end'])

        if overlap_start < overlap_end:
            confusion[(el1['text'], el2['text'])] += overlap_end - overlap_start

        if el1['end'] <= el2['end']:
            confusion[(el1['text'], "UNK")] += el1['end']- el1['start']
            index1 += 1
        else:
            confusion[(el2['text'], "UNK")] += el2['end']- el2['start']
            index2 += 1

    return dict(confusion)


def main():
    # Read golden alignemnts + curr alignments
    data = {}

    for orig in ['output','output_verified']:
        data[orig] = {} 
        for iso in ['shi','tzm']:
            data[orig][iso] = []
            for i in range(SAMPLES_TESTING):
                if i == 5 and iso == 'tzm':
                    continue
                filepath = f'./{orig}/{iso}/common_voice_22_0_{i}.TextGrid'
                data[orig][iso].append(load_intervals(filepath=filepath))

    def add_offset(interval,offset):
        interval['start'] += offset
        interval['end'] += offset
        return interval

    # Run per-character eval
    for iso in ['shi','tzm']:
        # We concatenate all sentences
        all_test = []
        all_pred = []
        offset = 0.
        for i in range(SAMPLES_TESTING - int(iso=='tzm')):
            test = data['output_verified'][iso][i]
            test = [add_offset(j,offset) for j in test]
            all_test += test
            pred = data['output'][iso][i]
            pred = [add_offset(j,offset) for j in pred]
            all_pred += pred
            # add offset last interval's end
            offset = max(all_test[-1]['end'],all_pred[-1]['end'])
        # We calculate confusion matrixes 
        pprint(confusion_matrix(all_test,all_pred))
        conf = confusion_matrix(all_test, all_pred )
        df = confusion_to_df(conf)
        matrix = to_matrix(df)
        matrix = matrix.drop(columns="UNK", errors="ignore")
        matrix = remove_diagonal(matrix)
        plot_confusion(matrix,iso)
        #pprint(all_test)
        #pprint(all_pred)
    assert(False)



    # Run per-sentence eval
    output = "iso,sentence,time_total,time_corr_abs,time_corr_perc\n"
    for orig in ['output','output_verified']:
        for iso in ['shi','tzm']:
            for i in range(SAMPLES_TESTING):
                correct,total = sentence_overlap(seq1, seq2)
                output += ",".join(iso,f"common_voice_22_0_{i}",total, correct, round(correct/total,3 ))




if __name__ == "__main__":
    main()
