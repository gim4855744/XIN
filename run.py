import pandas as pd
import numpy as np
import torch
import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, average_precision_score, accuracy_score

from preprocessor import Preprocessor
from model import XIN
from utils import set_global_seed, save_model, load_model

parser = argparse.ArgumentParser()

# global setting arguments
parser.add_argument('--mode', choices=('train', 'test'), type=str, required=True, help='select train or test')
parser.add_argument('--seed', type=int, help='global random seed for reproducibility')
parser.add_argument('--ckpt_path', default='checkpoint/model.ckpt', type=str, help='model checkpoint path')

# training setting arguments
parser.add_argument('--val_size', default=0.2, type=float, help='validation set size')

# dataset related arguments
parser.add_argument('--data_path', type=str, required=True, help='dataset file path')
parser.add_argument('--task', choices=('regression', 'binary_classfication', 'classification'), type=str, required=True, help='dataset task')

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def num_cat_split(df: pd.DataFrame):
    
    fields_names = df.columns

    fields = df[fields_names[:-1]]
    lable = df[fields_names[-1:]]

    numerical_fields_names = []
    categorical_fields_names = []
    num_categories_in_fields = []

    for column in fields.columns:
        if fields[column].dtype == 'object':
            categorical_fields_names.append(column)
            num_categories_in_fields.append(len(set(fields[column])))
        else:
            numerical_fields_names.append(column)

    return fields[numerical_fields_names], fields[categorical_fields_names], lable, num_categories_in_fields

def main():

    if args.seed is not None:
        set_global_seed(args.seed)

    df = pd.read_csv(args.data_path).dropna(axis=1)

    if args.mode == 'train':

        train_df, val_df = train_test_split(df, test_size=args.val_size)

        train_numerical_fields, train_categorical_fields, train_lable, num_categories_in_fields = num_cat_split(train_df)
        val_numerical_fields, val_categorical_fields, val_label, _ = num_cat_split(val_df)

        preprocessor = Preprocessor(args.task)
        train_numerical_fields, train_categorical_fields, train_lable = preprocessor.fit_transform(train_numerical_fields, train_categorical_fields, train_lable)
        val_numerical_fields, val_categorical_fields, val_label = preprocessor.transform(val_numerical_fields, val_categorical_fields, val_label)

        num_numerical_fields = train_numerical_fields.shape[-1]
        num_categorical_fields = train_categorical_fields.shape[-1]
        if args.task == 'classification':
            out_size = len(np.unique(train_lable))
        else:
            out_size = 1

        model = XIN(num_numerical_fields, num_categorical_fields, num_categories_in_fields, out_size, args.task).to(device)
        model.fit(
            train_numerical_fields, train_categorical_fields, train_lable,
            val_numerical_fields, val_categorical_fields, val_label
        )
        save_model(model, preprocessor, args.ckpt_path)

    elif args.mode == 'test':

        test_numerical_fields, test_categorical_fields, test_lable, _ = num_cat_split(df)

        model, preprocessor = load_model(args.ckpt_path)
        model = model.to(device)

        test_numerical_fields, test_categorical_fields, test_lable = preprocessor.transform(test_numerical_fields, test_categorical_fields, test_lable)

        predict = model.predict(test_numerical_fields, test_categorical_fields)

        if args.task == 'regression':
            mae = mean_absolute_error(test_lable, predict)
            rmse = np.sqrt(mean_squared_error(test_lable, predict))
            r2 = r2_score(test_lable, predict)
            print(f'MAE: {mae:.7f}, RMSE: {rmse:.7f}, R2: {r2:.7f}')
        elif args.task == 'binary_classification':
            auroc = roc_auc_score(test_lable, predict)
            auprc = average_precision_score(test_lable, predict)
            print(f'AUROC: {auroc:.7f}, AUPRC: {auprc:.7f}')
        else:
            predict = np.argmax(predict, axis=1)
            acc = accuracy_score(test_lable, predict)
            print(f'Accuracy: {acc:.7f}')

if __name__ == '__main__':
    main()
