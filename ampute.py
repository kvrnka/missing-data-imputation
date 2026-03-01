# файл с функциями для генерации пропусков
import numpy as np
import pandas as pd


def introduce_mcar(df, missing_rate=0.1, columns=None, random_state=None):
    """
    Вносит пропуски по механизму MCAR в DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Исходный датасет.
    missing_rate : float
        Доля пропусков (от 0 до 1).
    columns : list or None
        Список столбцов, в которых нужно создать пропуски.
        Если None — пропуски создаются во всех столбцах.
    random_state : int or None
        Seed для воспроизводимости.

    Returns
    -------
    pandas.DataFrame
        Новый DataFrame с пропусками.
    """
    if not 0 <= missing_rate <= 1:
        raise ValueError("missing_rate должен быть в диапазоне [0, 1].")

    df_mcar = df.copy()
    rng = np.random.default_rng(random_state)

    if columns is None:
        columns = df_mcar.columns
    else:
        missing_cols = set(columns) - set(df_mcar.columns)
        if missing_cols:
            raise ValueError(f"В датасете нет столбцов: {missing_cols}")

    # Генерируем маску только для выбранных столбцов
    mask = rng.uniform(size=(df_mcar.shape[0], len(columns))) < missing_rate

    df_mcar.loc[:, columns] = df_mcar.loc[:, columns].mask(mask)

    return df_mcar
