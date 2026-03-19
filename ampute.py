import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import root_mean_squared_error

def produce_na(data, target_col, dependency_col=None, mechanism='MCAR', ratio=0.2, random_state=42):
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
    
    # инициализируем маску (False = не стирать)
    mask = np.zeros(n, dtype=bool)

    # mcar: просто случайные индексы
    if mechanism == 'MCAR':
        missing_indices = np.random.choice(n, n_miss, replace=False)
        mask[missing_indices] = True

    # mar: вероятность зависит от другого столбца
    elif mechanism == 'MAR':
        if dependency_col is None:
            raise ValueError("Для MAR нужно указать dependency_col!")
            # берём зависимый столбец
        dep_values = df[dependency_col]
        
        # если категориальный -> кодируем
        if dep_values.dtype == 'object':
            le = LabelEncoder()
            dep_values = le.fit_transform(dep_values.astype(str))
        else:
            dep_values = dep_values.values
        
        # нормализация (0–1)
        dep_min = np.min(dep_values)
        dep_max = np.max(dep_values)
        
        # защита от деления на 0
        if dep_max - dep_min == 0:
            probs = np.ones(n) / n
        else:
            norm_dep = (dep_values - dep_min) / (dep_max - dep_min)
            
            # добавим небольшой шум
            noise = np.random.normal(0, 0.05, size=n)
            norm_dep = norm_dep + noise
            
            # убираем отрицательные значения
            norm_dep = np.clip(norm_dep, 0, None)
            
            # превращаем в вероятности
            if norm_dep.sum() == 0:
                probs = np.ones(n) / n
            else:
                probs = norm_dep / norm_dep.sum()
        
        # выбираем индексы с вероятностями
        mask = np.random.choice(n, size=n_miss, replace=False, p=probs)
        
    else:
        raise ValueError("Неизвестный механизм")

    # вставляем пропуски
    df.loc[mask, target_col] = np.nan
    
    return df, mask

def get_rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred, squared=False)
