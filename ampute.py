import numpy as np
import pandas as pd


def generate_missing_data(data, columns_config, mechanism='MCAR', ratio=0.2, random_state=42):
    """
    Создает пропуски в выбранных столбцах DataFrame по заданному механизму.

    Параметры:
    data (pd.DataFrame): исходный набор данных.
    columns_config (list[dict]): список словарей вида {target_col: dependency_col};
    для MCAR dependency_col не используется, для MAR используется для расчета вероятностей.
    mechanism (str): механизм генерации пропусков, поддерживаются MCAR и MAR.
    ratio (float): доля пропусков для каждого target_col (от 0 до 1).
    random_state (int): seed для воспроизводимости случайной генерации.

    Возвращает:
    tuple[pd.DataFrame, pd.DataFrame]:
    DataFrame с добавленными пропусками и булеву маску этих пропусков.
    """
    np.random.seed(random_state)
    df = data.copy()
    n = len(df)
    n_miss = int(n * ratio) # сколько точно значений стереть
    full_mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    for col_dict in columns_config:
        target_col = list(col_dict.keys())[0]
        dependency_col = list(col_dict.values())[0] if mechanism == "MAR" else None

        mask = np.zeros(n, dtype=bool)

        # MCAR
        if mechanism == 'MCAR':
            missing_indices = np.random.choice(n, n_miss, replace=False)
            mask[missing_indices] = True

        # MAR
        elif mechanism == 'MAR':
            if dependency_col is None:
                raise ValueError("Для MAR нужен dependency_col")

            dep_values = df[dependency_col]

            # кодируем категории
            if dep_values.dtype in ['object', 'string']:
                categories = dep_values.astype(str)
                unique_vals = categories.unique()
                # случайные веса для категорий
                weights = dict(zip(unique_vals, np.random.rand(len(unique_vals))))
                dep_values = categories.map(weights).values
            else:
                dep_values = dep_values.values

            # нормализация
            dep_min, dep_max = dep_values.min(), dep_values.max()

            if dep_max - dep_min == 0:
                probs = np.ones(n) / n
            else:
                norm_dep = (dep_values - dep_min) / (dep_max - dep_min)

                # шум
                norm_dep += np.random.normal(0, 0.05, size=n)
                # отбрасываем отрицательные значения после добавления шума
                norm_dep = np.clip(norm_dep, 0, None)
                # нормализация для получения вероятностей
                probs = norm_dep / norm_dep.sum() if norm_dep.sum() > 0 else np.ones(n) / n

            missing_indices = np.random.choice(n, size=n_miss, replace=False, p=probs)
            mask[missing_indices] = True

        else:
            raise ValueError("Unknown mechanism")

        # применяем пропуски
        df.loc[mask, target_col] = np.nan
        full_mask[target_col] = mask

    return df, full_mask
