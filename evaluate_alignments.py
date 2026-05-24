import tgt
import numpy as np
import chardet
from pprint import pprint

import sys # passing folder argument

from collections import defaultdict
from utils import SAMPLES_TESTING
import pandas as pd

import matplotlib.pyplot as plt
from numpy import var as var
from numpy import average as mean
from numpy import median as median
from scipy.stats import skew, kurtosis

MIN_SUPPORT = 8 # Min. Threshold of occurrances, to consider a pattern
PLOT_PATH = ""

def plot_histogram(data,filename):
    #pprint(data)
    data = [1000*x for x in data] #making things easier to compare with  https://aclanthology.org/2025.computel-main.11.pdf

    # statistics
    mean_val = np.mean(data)
    var_val = np.var(data)
    skew_val = skew(data)
    kurt_val = kurtosis(data)

    color_main = "steelblue"
    color_line = "red"
    plt.hist(data, bins=30, edgecolor='black',color=color_main )
    plt.axvline(0, color=color_line, linestyle='--', linewidth=2)
    plt.xlim(-200, 200)   # zoom in here
    plt.xlabel(f'{filename.upper()} Difference (gold labels-pred labels) in ms')
    plt.ylabel('Frequency')

    # stats text box
    stats_text = (
        f"Mean: {mean_val:.2f} ms\n"
        f"Variance: {var_val:.2f}\n"
        f"Skewness: {skew_val:.2f}\n"
        f"Kurtosis: {kurt_val:.2f}"
    )
    # place OUTSIDE graph
    plt.text(
        0.95, 0.95,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )


    plt.savefig(f"{PLOT_PATH}/{filename}_histogram_onset_dif.png")
    plt.clf()   # clear figure

def plot_barchart_persample(data):
    # Extract fields
    labels = [f"{iso.upper()}_{sample.split('_')[-1]}" for iso, sample, _, _, _ in data]
    filename = labels[0].split('_')[:-1]
    filename = '_'.join(filename)
    correct = [c for _, _, _, c, _ in data]
    total = [t for _, _, t, _, _ in data]
    pct = [p for _, _, _, _, p in data]
    # Plot
    plt.figure(figsize=(12, 6))
    x = range(len(data))
    plt.bar(x, correct, label='Correct')
    plt.bar(x, [t-c for t, c in zip(total, correct)], bottom=correct, label='Incorrect', alpha=0.5)
    
    avg_pct = round(sum(pct)/len(pct),3)

    var_pct = np.var([100*i for i in pct])  # population variance
    var_pct = round(var_pct,2)
    # Add percentage labels on bars
    for i, (c, t, p) in enumerate(zip(correct, total, pct)):
        plt.text(i, c/2, f"{p:.1%}", ha='center', va='center')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.xlabel('')  # optional: remove x-axis label if any
    plt.title(f"Success % of seconds per sample (avg= {avg_pct:.1%}, var={var_pct})")
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"{PLOT_PATH}/per_sample/{filename}_succes_per_sample.png")
    plt.clf()   # clear figure



def plot_matrix_heatmap(matrix,title = 'no_title',label='nolabel',filename='no_filename.png'):

    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, aspect="auto")

    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=90)
    plt.yticks(range(len(matrix.index)), matrix.index)

    plt.colorbar(label=label)
    plt.title(title)
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"{PLOT_PATH}/conf_mat/{filename}")
    plt.clf()   # clear figure



def load_intervals(filepath, word_level = False):
    intervals = []

    with open(filepath, 'rb') as f:
        encoding = chardet.detect(f.read())['encoding']
    tg = tgt.read_textgrid(filepath,encoding=encoding)
    if word_level:
        tier = tg.get_tier_by_name('words')  
    else:
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
    difs = []
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
                difs.append(el1['start']-el2['start'])
        else :
            if el1['end']<= el2['end']:
                time_total += el1['end']- el1['start']
            else: 
                time_total += el2['end']- el2['start']

        # Advance whichever interval ends first
        if el1['end']<= el2['end']:
            index1 += 1
        else:
            index2 += 1

    return time_correct, time_total, difs


def confusion_matrix(seq1, seq2):
    """Returns confusion matrix as dict of (sym1, sym2) -> time"""

    confusion = defaultdict(list)   # default 0.0
    index1 = index2 = 0
    while index1 < len(seq1) and index2 < len(seq2):
        el1 = seq1[index1]
        el2 = seq2[index2]

        overlap_start = max(el1['start'], el2['start'])
        overlap_end = min(el1['end'], el2['end'])

        if overlap_start < overlap_end:
            confusion[(el1['text'], el2['text'])].append(overlap_end - overlap_start)
        elif el1['end'] <= el2['end']:
            confusion[(el1['text'], "UNK")].append(el1['end']- el1['start'])
        else:
            confusion[("UNK",el2['text'])].append(el2['end']- el2['start'])

        if el1['end'] <= el2['end']:
            index1 += 1
        else:
            index2 += 1

    return dict(confusion)


def prepare_matrix(conf,remove_zeros = True, normalize = False,drop_unk=False):
    df = pd.DataFrame([
        {"ref": k[0], "hyp": k[1], "time": v}
        for k, v in conf.items()
    ])

    matrix = df.pivot_table(
        index="ref",
        columns="hyp",
        values="time",
        aggfunc="sum",
        fill_value=0
    )

    if drop_unk:
        matrix = matrix.drop(index='UNK', columns='UNK')

    if remove_zeros:
        matrix = matrix.loc[~(matrix == 0).all(1), ~(matrix == 0).all(0)]

    if normalize == 'row':
        matrix = matrix.div(matrix.sum(axis=1), axis=0)
    elif normalize == 'col':
        matrix = matrix.div(matrix.sum(axis=0), axis=1)

    return matrix



def f2s(x):
    return f"{x:.4f}"


def main():
    # Read golden alignemnts + curr alignments
    data = {}
    data_word= {}
    pred_folder  = sys.argv[1] if len(sys.argv) > 1 else "output"
    print(f'Prediction_folder is {pred_folder}')
    global PLOT_PATH
    PLOT_PATH = f"./plots/{pred_folder}"

    for orig in [pred_folder,'output_verified']:
        data[orig] = {} 
        data_word[orig] = {}
        for iso in ['shi','tzm']:
            data[orig][iso] = []
            data_word[orig][iso] = []
            for i in range(SAMPLES_TESTING):
                if (i == 5 or i== 0) and iso == 'tzm':
                    continue
                filepath = f'./{orig}/{iso}/common_voice_22_0_{i}.TextGrid'
                data[orig][iso].append(load_intervals(filepath=filepath))
                data_word[orig][iso].append(load_intervals(filepath=filepath,word_level=True))
    

    # Run per-sentence eval(and global)
    output = "iso,sentence,time_total,time_corr_abs,time_corr_perc\n"
    barchartdata = []
    histogramdata = {}
    histogramdata_nostop = {}
    for iso in ['shi','tzm']:
        histogramdata[iso]= {}
        histogramdata_nostop[iso]= {}
        for i in range(SAMPLES_TESTING - 2*int(iso=='tzm')): # we have one less sample in tzm
            if(iso == 'tzm' and (i == 5 or i == 0)):
                i+=1
                continue

            # Getting per-sentence metrics
            sample = f"common_voice_22_0_{i}"
            gold = data['output_verified'][iso][i]
            pred = data[pred_folder][iso][i]

            correct,total,difs = sentence_overlap(gold, pred)
            perc = round(correct/total,3)
            output += ",".join([f'{iso}_phon',sample,f2s(total), f2s(correct), f2s(perc)]) + '\n'
            barchartdata.append((f'{iso}_phon',sample,total, correct, round(correct/total,3)))

            gold = data_word['output_verified'][iso][i]
            pred = data_word[pred_folder][iso][i]

            correct,total,difs = sentence_overlap(gold, pred)
            perc = round(correct/total,3)
            output += ",".join([f'{iso}_word',sample,f2s(total), f2s(correct), f2s(perc)]) + '\n'
            barchartdata.append((f'{iso}_word',sample,total, correct, round(correct/total,3)))
            histogramdata[iso][i]= difs

            gold_nostop = [ x for x  in data_word['output_verified'][iso][i] if len(x['text']) > 2]
            pred_nostop = [ x for x  in data_word[pred_folder][iso][i] if len(x['text']) > 2]
            correct,total,difs = sentence_overlap(gold_nostop, pred_nostop)
            perc = round(correct/total,3)
            output += ",".join([f'{iso}_word_nostop',sample,f2s(total), f2s(correct), f2s(perc)]) + '\n'
            barchartdata.append((f'{iso}_word_nostop',sample,total, correct, round(correct/total,3)))
            histogramdata_nostop[iso][i] = difs
    for iso in ['shi', 'tzm']:
        tmp = []
        for k in histogramdata[iso]:
            tmp += histogramdata[iso][k] 
        plot_histogram(tmp, filename=f'{iso}_all')
        tmp = []
        for k in histogramdata_nostop[iso]:
            tmp += histogramdata_nostop[iso][k] 
        plot_histogram(tmp, filename=f'{iso}_nostop')

    


    with open(f"{PLOT_PATH}/per_sample/all.csv", "w") as f:
        f.write(output)
    for iso in ['shi','tzm']:
        for cat in ['phon','word','word_nostop']:
            label = f'{iso}_{cat}'
            subset_data = [(l,a,b,c,d) for (l,a,b,c,d) in barchartdata if l ==label]
            plot_barchart_persample(data=subset_data)


    # Defining it here since I use it as a lambda function
    def add_offset(interval,offset):
        interval['start'] += offset
        interval['end'] += offset
        return interval


    # Run per-phone eval
    for iso in ['shi','tzm']:
        # We concatenate all sentences
        all_test = []
        all_pred = []
        offset = 0.
        for i in range(SAMPLES_TESTING - 2*int(iso=='tzm')): # we have one less sample in tzm
            test = data['output_verified'][iso][i]
            test = [add_offset(j,offset) for j in test]
            all_test += test
            pred = data[pred_folder][iso][i]
            pred = [add_offset(j,offset) for j in pred]
            all_pred += pred
            # new offset is last interval's end
            offset = max(all_test[-1]['end'],all_pred[-1]['end'])


        for func_name in ["size","mean","median","var"]:
            conf = confusion_matrix(all_test, all_pred)
            for k in conf.keys():
                #print(conf[k])
                if len(conf[k]) < MIN_SUPPORT:
                    conf[k]= 0
                else:
                    conf[k] = getattr(np, func_name)(conf[k])
            for drop_unk in False,True:
                for normalize in ['row','col',False]:
                    df = prepare_matrix(conf, drop_unk=drop_unk,normalize=normalize)
                    title = f"Confusion Matrix {iso.upper()} (Golden label X/Pred Y)"
                    if normalize:
                        title += f" normalized by {normalize}"
                    label = "%(of seconds)" if normalize else 'time(seconds)'
                    if func_name == "size":
                        label =  "Amount"
                    unk_txt = 'nounk' if drop_unk else 'unk'
                    norm_txt = f'norm{normalize}' if normalize else 'normnone'
                    filename = f"{iso}_{func_name}_{unk_txt}_{norm_txt}.png"
                    plot_matrix_heatmap(df, title = title, label=label, filename=filename)


        # Pairs descending by variance
        conf = confusion_matrix(all_test, all_pred)
        conf = [(gold,pred,f2s(sum(conf[(gold,pred)])),f2s(len(conf[(gold,pred)])),f2s(mean(conf[(gold,pred)])),f2s(var(conf[(gold,pred)]))) for (gold,pred) in conf.keys()]
        conf = list(sorted(conf,key=lambda x: int(float(x[3])),reverse=True))#[:15]
        with open(f"{PLOT_PATH}/conf_mat/{iso}_unfiltered.csv", "w") as f:
            output = "gold,pred,time_sec,support,mean, median,variance\n"
            output += '\n'.join([','.join(l) for l in conf])
            f.write(output)
    


if __name__ == "__main__":
    main()
