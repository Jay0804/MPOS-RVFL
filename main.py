import csv
import numpy as np
import warnings
import os
import time
import pandas as pd
import skmultiflow as sk
from sklearn.metrics import accuracy_score
from utils import para_init
warnings.filterwarnings("ignore")

def cal_Gmeans(y_real, y_pred):
    """

    :param y_real: real labels
    :param y_pred: predict labels
    :return: G-mean value
    """
    fault_types, count_types = np.unique(y_real, return_counts=True)
    num_type = len(fault_types)
    correct_types = np.zeros(num_type)
    acc_types = np.zeros(num_type)
    gmeans = 1
    for p in range(len(y_pred)):
        if y_real[p] == y_pred[p]:
            correct_types[int(y_real[p])] += 1
    for p in range(num_type):
        acc_types[p] = correct_types[p] / count_types[p]
        gmeans = gmeans * acc_types[p]
    # demo is a 3 classification problem
    gmeans = np.cbrt(gmeans)
    return gmeans


start_time = time.time()  # Record the program start time
batchsize = 5
n_anchor = 50
n_round = 2
n_class = 3

n_ratios = np.arange(5, 100, 5).tolist()

clf_name_list_all = [['MPOSRVFL']]

for clf_name_list in clf_name_list_all:
    num_clf = len(clf_name_list)
    num_ratio = len(n_ratios)

    acc_list = [[[[] for _ in range(n_round)] for _ in range(num_clf)] for _ in range(num_ratio)]
    gmeans_list = [[[[] for _ in range(n_round)] for _ in range(num_clf)] for _ in range(num_ratio)]

    for ratio_index, ratio in enumerate(n_ratios):
        ratio = ratio if isinstance(ratio, int) else ratio[0]
        # Data loading
        file_name = './imbalance_data/'
        data_off = np.array(pd.read_csv(f"{file_name}data_off_type({n_class})_ratio({ratio}).csv", header=None))
        label_off = np.array(pd.read_csv(f"{file_name}label_off_type({n_class})_ratio({ratio}).csv", header=None)).flatten()
        data_stream = np.array(pd.read_csv(f"{file_name}data_stream_type({n_class})_ratio({ratio}).csv", header=None))
        label_stream = np.array(pd.read_csv(f"{file_name}label_stream_type({n_class})_ratio({ratio}).csv", header=None)).flatten()

        # Result Record
        max_samples = len(data_stream)

        directory_path = f'./result_type({n_class})_batchsize({batchsize})_nRatios({len(n_ratios)})_{clf_name_list[0]}/'
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        for n_clf in range(len(clf_name_list)):
            clf_name = clf_name_list[n_clf]
            para_clf = para_init(n_anchor=n_anchor)

            y_pred_all = []
            y_true_all = []
            t1 = time.time()
            round_path = directory_path + f'Gmean_per_round/'
            if not os.path.exists(round_path):
                os.makedirs(round_path)

            for round in range(n_round):
                clf = para_clf.get_clf("clf_" + clf_name)
                y_pred_list = []
                y_true_list = []

                'stream initialization'
                stream = sk.data.DataStream(data_stream, label_stream)
                count = 0

                # Pretrain
                clf.fit(data_off, label_off)

                # Train the classifier with the samples provided by the data stream
                while stream.has_more_samples():
                    X, y = stream.next_sample(batchsize)

                    count += len(X)
                    if count % 100 == 0:
                        print(count, '/', max_samples)

                    y_pred = clf.predict(X)
                    y_pred_list.extend(y_pred)
                    y_true_list.extend(y)

                    clf.partial_fit(X, y)
                gmeans_list[ratio_index][n_clf][round] = cal_Gmeans(y_true_list, y_pred_list)
                acc_list[ratio_index][n_clf][round] = accuracy_score(y_true_list, y_pred_list)

                y_pred_all = y_pred_all + y_pred_list
                y_true_all = y_true_all + y_true_list

            t2 = time.time()
            result_pred = np.array(y_pred_all).reshape(n_round, count)
            result_true = np.array(y_true_all).reshape(n_round, count)

            directory_path_result = directory_path + f'Predict_result/'
            if not os.path.exists(directory_path_result):
                os.makedirs(directory_path_result)
            result_pred_name = directory_path_result + f'Prediction_ratio({ratio}).csv'
            result_true_name = directory_path_result + f'True_ratio({ratio}).csv'

            with open(result_pred_name, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(result_pred)
            with open(result_true_name, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(result_true)

            print("\nAccuracy %s + Ratio(%s): %.3f $\pm$ %.3f" % (
                clf_name_list[n_clf], ratio, np.mean(acc_list[ratio_index][n_clf]),
                np.std(acc_list[ratio_index][n_clf])))
            print("Gmeans %s + Ratio(%s): %.3f $\pm$ %.3f" %
                  (clf_name_list[n_clf], ratio, np.mean(gmeans_list[ratio_index][n_clf]),
                   np.std(gmeans_list[ratio_index][n_clf])))
            print("Average Time %s + %s: %.4f s\n" % (clf_name_list[n_clf], ratio, (t2 - t1) / n_round))

    # Result
    print('----------------- show G-mean result --------------------')
    print(f"ratios = {n_ratios}")
    for i in range(num_clf):
        gmeans_list = np.array(gmeans_list)  # (num_ratio, n_clf, n_round)
        gmean_mean = np.mean(gmeans_list, axis=2)
        print(f"G-mean {clf_name_list[i]}: {gmean_mean.T[i]}")

    for i in range(num_clf):
        gmeans_list = np.array(gmeans_list)  # (num_ratio, n_clf, n_round)
        gmean_std = np.std(gmeans_list, axis=2)

    # --------------------------------------------- Data Saving -----------------------------------------------------
    file_name = directory_path
    file_path = file_name + f'Gmean_average.csv'
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(clf_name_list)
        writer.writerows(gmean_mean)
    print(f"Data was successfully written to {file_path}")

    file_path = file_name + f'Gmean_std.csv'
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(clf_name_list)
        writer.writerows(gmean_std)
    print(f"Data was successfully written to {file_path}")

    file_path = file_name + f'Ratios.csv'
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['n_ratios'])
        n_ratios = np.array(n_ratios).reshape(-1, 1)
        writer.writerows(n_ratios)
    print(f"Data was successfully written to {file_path}")

    for round in range(n_round):
        file_path = round_path + f'Gmean_round_{round}.csv'
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(clf_name_list)
            writer.writerows(gmeans_list[:, :, round])
        print(f"Data was successfully written to {file_path}")

    end_time = time.time()  #
    elapsed_time = end_time - start_time  # Calculation time (in seconds)
    print(f"Program run time：{elapsed_time:.2f} 秒")
