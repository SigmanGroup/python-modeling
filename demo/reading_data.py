import pandas as pd
import numpy as np

def read_single_file(excel_file="../model_oct_bothsubs", excel_sheet="bubbleplot", header_row=4, num_par=2, par_start_col=9, num_samples=24, response_col=5, y_label_col = 0, apply_mask=True, verbose=True, xlabelrow=True):

    inp = pd.read_excel(excel_file + ".xlsx", excel_sheet, header=header_row, index_col=y_label_col, nrows=num_samples + int(xlabelrow), usecols=list(range(0, (num_par + par_start_col))))
    print(inp.head())
    print()

    if xlabelrow:
        X_names = list(inp.iloc[0, par_start_col - 1:num_par + par_start_col - 1])
        X_labels = list(inp.columns)[par_start_col - 1:num_par + par_start_col - 1]
        resp_label = list(inp.columns)[response_col - 1]
        inp.drop(index=inp.index[0], inplace=True)
    else:
        X_labels = list(inp.columns)[par_start_col - 1:num_par + par_start_col - 1]
        X_names = X_labels
        resp_label = list(inp.columns)[response_col - 1]

    X_labelname = [" ".join(i) for i in zip(X_labels, X_names)]
    X_labelname_dict = dict(zip(X_labels, X_names))
    y = np.asarray(inp[resp_label], dtype=float)
    X = np.asarray(inp[X_labels], dtype=float)
    y_labels = np.asarray(list(inp.index), dtype=str)
    y_labels_comp = y_labels

    if apply_mask:
        mask = y.nonzero()[0]
        mask = ~np.isnan(y)
        print("n_samples before removing empty cells: {}".format(len(y)))
        print("Removing {} samples.".format(len(y) - sum(mask)))
        X = X[np.array(mask)]
        y = y[np.array(mask)]
        y_labels = y_labels[np.array(mask)]
    X_all = X
    if verbose:
        print("Shape X: {}".format(X.shape))
        print("Shape y: {}".format(y.shape))
        print("Shape labels: {}".format(y_labels.shape))
        print("First X cell: {}".format(X[0, 0]))
        print("Last X cell:  {}".format(X[-1, -1]))
        print("First y: {}".format(y[0]))
        print("Last y:  {}".format(y[-1]))
        print("Last label: {}".format(y_labels[-1]))

