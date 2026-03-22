import time
from imputation import imputation
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import root_mean_squared_error


def rmse_std(y_true, y_pred):
    mean = y_true.mean()
    std = y_true.std()
    
    if std == 0:
        return 0.0
    
    y_true_std = (y_true - mean) / std
    y_pred_std = (y_pred - mean) / std
    
    return root_mean_squared_error(y_true_std, y_pred_std)


def cat_accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()


def load_dataset(name):
    # путь относительно файла, где лежит функция
    base_path = Path(__file__).parent  # папка скрипта
    file_path = base_path / "data/processed" / name / "ground_truth.csv"
    
    df = pd.read_csv(file_path)
    return df


def run_single_experiment(df, df_incomplete, mask, method, params, columns):
    """
    df: исходный датасет без пропусков
    df_incomplete: датасет с пропусками
    mask: булев DataFrame, где True = пропуск
    method: строка с названием метода импутации
    params: параметры для метода
    columns: список колонок для оценки
    """
    start = time.perf_counter()

    # импутация
    imputed_data = imputation(
        df_incomplete=df_incomplete,
        algo=method,
        **params
    )

    end = time.perf_counter()

    # определяем типы колонок
    num_cols = df[columns].select_dtypes(include=[float, int]).columns
    cat_cols = df[columns].select_dtypes(include=['object', 'category', 'string']).columns

    # rmse для числовых
    rmse_score = None
    if len(num_cols) > 0:
        rmse_list = []
        for col in num_cols:
            y_true_col = df.loc[mask[col], col]
            y_pred_col = imputed_data.loc[mask[col], col]
            rmse_list.append(rmse_std(y_true_col, y_pred_col))
        rmse_score = sum(rmse_list) / len(rmse_list)

    # accuracy для категориальных
    acc_score = None
    if len(cat_cols) > 0:
        acc_list = []
        for col in cat_cols:
            y_true_col = df.loc[mask[col], col]
            y_pred_col = imputed_data.loc[mask[col], col]
            acc_list.append(cat_accuracy(y_true_col, y_pred_col))
        acc_score = sum(acc_list) / len(acc_list)

    return rmse_score, acc_score, end - start


def log_experiment(
    experiment,
    dataset_name,
    mechanism,
    method,
    params,
    ratio,
    rmse=None,
    acc=None,
    time_taken=None
):
    """
    Логирует эксперимент с понятными и структурированными именами метрик
    """

    # превращаем параметры в строку
    params_str = "_".join([f"{k}={v}" for k, v in sorted(params.items())]) if params else "default"

    # базовый префикс
    base_name = (
        f"{dataset_name}"
        f"|mech={mechanism}"
        f"|method={method}"
        f"|params={params_str}"
        f"|ratio={round(ratio, 2)}"
    )

    step = int(ratio * 100)

    # RMSE
    if rmse is not None:
        experiment.log_metric(
            name=f"{base_name}|metric=rmse",
            value=rmse,
            step=step
        )

    # Accuracy
    if acc is not None:
        experiment.log_metric(
            name=f"{base_name}|metric=accuracy",
            value=acc,
            step=step
        )

    # Time
    if time_taken is not None:
        experiment.log_metric(
            name=f"{base_name}|metric=time",
            value=time_taken,
            step=step
        )
