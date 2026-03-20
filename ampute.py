import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def generate_missing_data(data, columns_config, mechanism='MCAR', ratio=0.2, random_state=42):
    """
    Генерирует пропуски в столбце target_col.
    
    data: DataFrame
    target_col: имя столбца, где будут дырки
    dependency_col: имя столбца, от которого зависит пропуск (для MAR)
    mechanism: 'MCAR', 'MAR', 'MNAR'
    ratio: доля пропусков (0.2 = 20%)
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

        # --- MCAR ---
        if mechanism == 'MCAR':
            missing_indices = np.random.choice(n, n_miss, replace=False)
            mask[missing_indices] = True

        # --- MAR ---
        elif mechanism == 'MAR':
            if dependency_col is None:
                raise ValueError("Для MAR нужен dependency_col")

            dep_values = df[dependency_col]

            # кодируем категории
            if dep_values.dtype in ['object', 'string']:
                categories = dep_values.astype(str)
                unique_vals = categories.unique()
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
                norm_dep = np.clip(norm_dep, 0, None)

                probs = norm_dep / norm_dep.sum() if norm_dep.sum() > 0 else np.ones(n) / n

            missing_indices = np.random.choice(n, size=n_miss, replace=False, p=probs)
            mask[missing_indices] = True

        else:
            raise ValueError("Unknown mechanism")

        # применяем пропуски
        df.loc[mask, target_col] = np.nan
        full_mask[target_col] = mask

    return df, full_mask
