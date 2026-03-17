import pandas as pd

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

# from sklearn.preprocessing import OrdinalEncoder

# TODO сразу в функции добавить encoder

def get_rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)


def simple_imputation(df_incomplete, strategy = 'mean'):
    """
    Заполняет пропуски простой статистикой.
    strategy: 'mean' (среднее) или 'median' (медиана)
    """
    imputer = SimpleImputer(strategy=strategy)
    data_imputed = imputer.fit_transform(df_incomplete)
    return pd.DataFrame(data_imputed, columns=df_incomplete.columns, index=df_incomplete.index)


def knn_imputation(df_incomplete, n_neighbors=5):
    """
    Заполняет пропуски методом k-ближайших соседей.
    Автоматически масштабирует данные перед работой.
    """
    # масштабируем данные
    scaler = StandardScaler()
    df_scaled = df_incomplete.copy()
    # fit_transform вернет массив, возвращаем обратно в DataFrame
    df_scaled = pd.DataFrame(scaler.fit_transform(df_scaled), 
                             columns=df_incomplete.columns, index=df_incomplete.index)
    
    # запускаем KNN
    imputer = KNNImputer(n_neighbors=n_neighbors)
    data_imputed_scaled = imputer.fit_transform(df_scaled)
    
    # возвращаем масштаб обратно
    data_imputed = scaler.inverse_transform(data_imputed_scaled)
    
    return pd.DataFrame(data_imputed, columns=df_incomplete.columns, index=df_incomplete.index)


IMPUTATION_METHODS = {
    "Simple": simple_imputation,
    "KNN": knn_imputation,
    # limit_direction='both' позволяет заполнять пропуски в начале и в конце ряда, а не только между значениями
    "LinearInterpolation": lambda df, method: df.interpolate(method=method, limit_direction='both')
}

def imputation(df_incomplete, algo='Simple', **kwargs):
    # TODO сделать описание функции
    if algo not in IMPUTATION_METHODS:
        raise ValueError(f"Unknown mechanism: {algo}")
    
    return IMPUTATION_METHODS[algo](df_incomplete, **kwargs)
    