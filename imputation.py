import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer, KNNImputer

from fancyimpute import IterativeImputer

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import root_mean_squared_error


# def get_rmse(y_true, y_pred):
#     return root_mean_squared_error(y_true, y_pred)


def simple_imputation(df_incomplete, num_strategy='mean'):
    """
    Заполняет пропуски в датафрейме.
    
    Числовые колонки: num_strategy ('mean' или 'median')
    Категориальные колонки: заполняются самым частым значением
    """
    df = df_incomplete.copy()
    
    # определяем числовые и категориальные колонки
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns
    
    # импутация числовых
    if len(num_cols) > 0:
        num_imputer = SimpleImputer(strategy=num_strategy)
        df[num_cols] = num_imputer.fit_transform(df[num_cols])
    
    # импутация категориальных
    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    
    return df


def knn_imputation(df_incomplete, n_neighbors=5):
    """
    Заполняет пропуски методом k-ближайших соседей для числовых и категориальных колонок.
    Числовые колонки масштабируются перед KNN.
    Категориальные колонки кодируются OrdinalEncoder.
    """
    df = df_incomplete.copy()
    
    # разделяем числовые и категориальные
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns
    
    # числовые колонки
    if len(num_cols) > 0:
        scaler = StandardScaler()
        num_scaled = pd.DataFrame(scaler.fit_transform(df[num_cols]),
                                  columns=num_cols, index=df.index)
    else:
        num_scaled = pd.DataFrame(index=df.index)
    
    # категориальные колонки
    if len(cat_cols) > 0:
        enc = OrdinalEncoder()
        cat_encoded = pd.DataFrame(enc.fit_transform(df[cat_cols].astype(str)),
                                   columns=cat_cols, index=df.index)
    else:
        cat_encoded = pd.DataFrame(index=df.index)
    
    # объединяем все колонки в один датафрейм для KNN
    df_knn_input = pd.concat([num_scaled, cat_encoded], axis=1)
    
    # KNN Imputation
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed_array = imputer.fit_transform(df_knn_input)
    
    # разделяем обратно на числовые и категориальные
    df_imputed = pd.DataFrame(df_imputed_array, columns=df_knn_input.columns, index=df.index)
    
    # обратное масштабирование числовых
    if len(num_cols) > 0:
        df_imputed[num_cols] = scaler.inverse_transform(df_imputed[num_cols])
    
    # обратное преобразование категориальных
    if len(cat_cols) > 0:
        df_imputed[cat_cols] = np.round(df_imputed[cat_cols])  # округляем до целых
        df_imputed[cat_cols] = enc.inverse_transform(df_imputed[cat_cols])
    
    return df_imputed


def mice_imputation(df_incomplete, random_state=42, **kwargs):
    """
    MICE-импутация для смешанных данных (числовые + категориальные)
    df_incomplete: DataFrame с пропусками
    random_state: для воспроизводимости
    kwargs: любые дополнительные параметры для IterativeImputer
    """
    df = df_incomplete.copy()
    
    # разделяем типы
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns
    
    # кодируем категориальные в числа
    if len(cat_cols) > 0:
        enc = OrdinalEncoder()
        df[cat_cols] = enc.fit_transform(df[cat_cols].astype(str))
    else:
        enc = None
    
    # применяем IterativeImputer
    imputer = IterativeImputer(random_state=random_state, **kwargs)
    imputed_array = imputer.fit_transform(df)
    
    df_imputed = pd.DataFrame(imputed_array, columns=df.columns, index=df.index)
    
    # обратное преобразование категориальные
    if enc is not None and len(cat_cols) > 0:
        df_imputed[cat_cols] = np.round(df_imputed[cat_cols])
        df_imputed[cat_cols] = enc.inverse_transform(df_imputed[cat_cols])
    
    return df_imputed


IMPUTATION_METHODS = {
    "Simple": simple_imputation,
    "KNN": knn_imputation,
    "MICE": mice_imputation,
}

def imputation(df_incomplete, algo='Simple', **kwargs):
    # TODO сделать описание функции
    if algo not in IMPUTATION_METHODS:
        raise ValueError(f"Unknown mechanism: {algo}")
    
    return IMPUTATION_METHODS[algo](df_incomplete, **kwargs)
    