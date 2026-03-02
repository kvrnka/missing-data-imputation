import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error

def produce_na(data, target_col, dependency_col=None, mechanism='MCAR', ratio=0.2):
    """
    Генерирует пропуски в столбце target_col.
    
    data: DataFrame
    target_col: имя столбца, где будут дырки
    dependency_col: имя столбца, от которого зависит пропуск (для MAR)
    mechanism: 'MCAR', 'MAR', 'MNAR'
    ratio: доля пропусков (0.2 = 20%)
    """
    df = data.copy()
    n = len(df)
    n_miss = int(n * ratio) # сколько точно значений стереть
    
    # инициализируем маску (False = не стирать)
    mask = np.zeros(n, dtype=bool)

    # mcar: просто случайные индексы
    if mechanism == 'MCAR':
        missing_indices = np.random.choice(n, n_miss, replace=False)
        mask[missing_indices] = True

    # mnar: вероятность зависит от самого значения
    elif mechanism == 'MNAR':
        # чем больше значение, тем больше вероятность пропуска
        # (имитация: богатые скрывают доход)
        values = df[target_col].values
        
        # нормируем вероятность (добавляем немного шума, чтобы не было жесткой отсечки)
        probs = values + np.random.normal(0, values.std() * 0.1, size=n)
        
        # получаем индексы n_miss самых больших значений probabilities
        missing_indices = np.argsort(probs)[-n_miss:]
        mask[missing_indices] = True

    # mar: вероятность зависит от другого столбца
    elif mechanism == 'MAR':
        if dependency_col is None:
            raise ValueError("Для MAR нужно указать dependency_col!")
            
        # берем значения соседнего столбца
        dep_values = df[dependency_col].values
        
        # чем выше значение в dependency_col, тем выше шанс пропажи в target_col
        # добавляем шум для реалистичности
        probs = dep_values + np.random.normal(0, dep_values.std() * 0.1, size=n)
        
        missing_indices = np.argsort(probs)[-n_miss:]
        mask[missing_indices] = True
        
    else:
        raise ValueError("Неизвестный механизм")

    # вставляем пропуски
    df.loc[mask, target_col] = np.nan
    
    return df, mask

def get_rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred, squared=False)
