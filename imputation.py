import pandas as pd
import numpy as np
import gower

from scipy.stats import mode

from sklearn.impute import SimpleImputer, KNNImputer

from fancyimpute import IterativeImputer

from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn.metrics import root_mean_squared_error

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


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


def knn_imputation_mixed_data(df_incomplete, n_neighbors=5):
    df = df_incomplete.copy()
    df_for_gower = df.copy()

    # --- fix типов для gower ---
    for col in df_for_gower.select_dtypes(include='string').columns:
        df_for_gower[col] = df_for_gower[col].astype(object)

    # --- нормализация числовых ---
    num_cols = df_for_gower.select_dtypes(include=np.number).columns

    df_norm = df_for_gower.copy()
    df_norm[num_cols] = (df_norm[num_cols] - df_norm[num_cols].min()) / (
        df_norm[num_cols].max() - df_norm[num_cols].min()
    )

    df_imputed = df.copy()

    # --- KNN по колонкам ---
    for col in df.columns:
        missing_idx = df[df[col].isna()].index

        if len(missing_idx) == 0:
            continue

        valid_idx = df[~df[col].isna()].index

        # --- исключаем текущую колонку ---
        X_missing = df_norm.loc[missing_idx].drop(columns=[col])
        X_valid = df_norm.loc[valid_idx].drop(columns=[col])

        # --- считаем расстояния ---
        distances = gower.gower_matrix(X_missing, X_valid)

        for i, row_idx in enumerate(missing_idx):
            dists = distances[i]

            neighbors = np.argsort(dists)
            k_neighbors = valid_idx[neighbors[:n_neighbors]]

            values = df.loc[k_neighbors, col]

            # --- импутация ---
            if pd.api.types.is_numeric_dtype(df[col]):
                df_imputed.at[row_idx, col] = values.mean()
            else:
                if not values.mode().empty:
                    df_imputed.at[row_idx, col] = values.mode().iloc[0]

    return df_imputed


def knn_imputation_numeric_data(df_incomplete, n_neighbors=5):
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


# TODO улучшить гибрид, чтобы knn для числовых смотрел и на категориальные
def knn_imputation_hybrid(df_incomplete, n_neighbors=5):
    df = df_incomplete.copy()

    # --- 1. делим признаки ---
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(exclude=np.number).columns

    df_imputed = df.copy()

    # =========================
    # ЧИСЛОВЫЕ — sklearn KNN
    # =========================
    if len(num_cols) > 0:
        scaler = StandardScaler()
        df_num = df[num_cols]

        df_scaled = pd.DataFrame(
            scaler.fit_transform(df_num),
            columns=num_cols,
            index=df.index
        )

        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_scaled = imputer.fit_transform(df_scaled)

        df_imputed[num_cols] = scaler.inverse_transform(imputed_scaled)

    # =========================
    # КАТЕГОРИИ — Gower KNN
    # =========================
    if len(cat_cols) > 0:
        df_for_gower = df.copy()

        # фиксим типы
        for col in df_for_gower.select_dtypes(include='string').columns:
            df_for_gower[col] = df_for_gower[col].astype(object)

        # нормализация числовых (для расстояния)
        if len(num_cols) > 0:
            df_norm = df_for_gower.copy()
            denom = (df_norm[num_cols].max() - df_norm[num_cols].min()).replace(0, 1)
            df_norm[num_cols] = (df_norm[num_cols] - df_norm[num_cols].min()) / denom
        else:
            df_norm = df_for_gower.copy()

        # --- KNN по категориям ---
        for col in cat_cols:
            missing_idx = df[df[col].isna()].index

            if len(missing_idx) == 0:
                continue

            valid_idx = df[~df[col].isna()].index

            X_missing = df_norm.loc[missing_idx].drop(columns=[col])
            X_valid = df_norm.loc[valid_idx].drop(columns=[col])

            distances = gower.gower_matrix(X_missing, X_valid)

            for i, row_idx in enumerate(missing_idx):
                dists = distances[i]

                neighbors = np.argsort(dists)
                k_neighbors = valid_idx[neighbors[:n_neighbors]]

                values = df.loc[k_neighbors, col]

                if not values.mode().empty:
                    df_imputed.at[row_idx, col] = values.mode().iloc[0]

    return df_imputed


# TODO сделать ранний критерий остановки для MICE
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


def custom_mice_imputation(df_incomplete, max_iter=10, random_state=42):
    np.random.seed(random_state)
    df_raw = df_incomplete.copy()
    df = df_raw.copy()

    # --- 1. типы колонок ---
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    df[cat_cols] = df[cat_cols].astype(object)

    # --- 2. LabelEncoder для категориальных ---
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        not_null = df[col].notna()
        df.loc[not_null, col] = le.fit_transform(df.loc[not_null, col].astype(str))
        encoders[col] = le

    # --- 3. начальная простая импутация ---
    df_imputed = df.copy()
    for col in num_cols:
        df_imputed[col] = df[col].fillna(df[col].mean())
    for col in cat_cols:
        mode = df[col].mode()
        if not mode.empty:
            df_imputed[col] = df[col].fillna(mode.iloc[0])

    # --- 4. OneHotEncoder для признаков ---
    ohe = None
    if cat_cols:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        ohe.fit(df_raw[cat_cols].astype(str))

    # --- 5. MICE итерации ---
    for _ in range(max_iter):
        for col in df.columns:
            missing_mask = df[col].isna()
            if missing_mask.sum() == 0:
                continue

            # --- target ---
            y = df_imputed[col]

            # --- признаки ---
            X = df_imputed.drop(columns=[col])
            X_num = X[[c for c in num_cols if c != col]].copy()

            # категориальные кроме текущей
            if cat_cols:
                # --- внутри MICE итераций ---
                cat_features = [c for c in cat_cols if c != col]
                if cat_features:
                    X_cat = X[cat_features].astype(str)
                    ohe_local = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                    ohe_local.fit(X_cat)  # фитим только на текущие признаки
                    X_cat_ohe = pd.DataFrame(ohe_local.transform(X_cat), index=X_cat.index)
                    X_full = pd.concat([X_num, X_cat_ohe], axis=1)
                else:
                    X_full = X_num
            else:
                X_full = X_num

            X_full.columns = X_full.columns.astype(str)

            X_train = X_full.loc[~missing_mask]
            y_train = y.loc[~missing_mask]
            X_missing = X_full.loc[missing_mask]

            # --- масштабирование числовых признаков ---
            scaler = StandardScaler()
            num_in_X = [c for c in X_train.columns if c in X_num.columns]
            if num_in_X:
                X_train[num_in_X] = scaler.fit_transform(X_train[num_in_X])
                X_missing[num_in_X] = scaler.transform(X_missing[num_in_X])

            # --- модель ---
            if col in num_cols:
                model = LinearRegression()
                model.fit(X_train, y_train)
                preds = model.predict(X_missing)
            else:
                y_train = y_train.astype(int)  # Явное приведение типов для классификатора
                if y_train.nunique() < 2:
                    continue
                # model = LogisticRegression(solver='lbfgs', max_iter=200, multi_class='multinomial')
                model = LogisticRegression(solver='lbfgs', max_iter=200)
                model.fit(X_train, y_train)
                preds = model.predict(X_missing)

            df_imputed.loc[missing_mask, col] = preds

    # --- 6. обратное преобразование категориальных ---
    for col in cat_cols:
        le = encoders[col]
        df_imputed[col] = df_imputed[col].round().astype(int)
        df_imputed[col] = le.inverse_transform(df_imputed[col])

    return df_imputed


def missforest_imputation(df_incomplete, max_iter=10, n_estimators=100, random_state=42):
    """
    Реализация алгоритма MissForest (MICE на базе случайных лесов).
    Для числовых признаков - RandomForestRegressor.
    Для категориальных - RandomForestClassifier.
    """
    np.random.seed(random_state)
    df_raw = df_incomplete.copy()
    df = df_raw.copy()

    # --- 1. типы колонок ---
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    df[cat_cols] = df[cat_cols].astype(object)

    # --- 2. LabelEncoder для категориальных (с сохранением NaN!) ---
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        not_null = df[col].notna()
        # кодируем только заполненные значения, NaN остаются NaN (иначе модель не увидит пропуски!)
        df.loc[not_null, col] = le.fit_transform(df.loc[not_null, col].astype(str))
        encoders[col] = le

    # --- 3. начальная простая импутация (mean / mode) ---
    df_imputed = df.copy()
    for col in num_cols:
        df_imputed[col] = df[col].fillna(df[col].mean())
    for col in cat_cols:
        mode = df[col].mode()
        if not mode.empty:
            df_imputed[col] = df[col].fillna(mode.iloc[0])

    # --- 4. Итерации MissForest ---
    for _ in range(max_iter):
        for col in df.columns:
            missing_mask = df[col].isna()
            if missing_mask.sum() == 0:
                continue

            # --- target ---
            y = df_imputed[col]
            # --- признаки (без OHE, деревья работают напрямую с label encoding) ---
            X = df_imputed.drop(columns=[col])

            X_train = X.loc[~missing_mask]
            y_train = y.loc[~missing_mask]
            X_missing = X.loc[missing_mask]

            # --- модель ---
            if col in num_cols:
                model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
                model.fit(X_train, y_train)
                preds = model.predict(X_missing)
            else:
                y_train = y_train.astype(int)
                if y_train.nunique() < 2:
                    continue
                model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
                model.fit(X_train, y_train)
                preds = model.predict(X_missing)

            df_imputed.loc[missing_mask, col] = preds

    # --- 5. обратное преобразование категориальных ---
    for col in cat_cols:
        le = encoders[col]
        df_imputed[col] = df_imputed[col].round().astype(int)
        df_imputed[col] = le.inverse_transform(df_imputed[col])

    return df_imputed


IMPUTATION_METHODS = {
    "Simple": simple_imputation,
    "KNN": knn_imputation_hybrid,
    "MICE": custom_mice_imputation,
    "MissForest": missforest_imputation
}

def imputation(df_incomplete, algo='Simple', **kwargs):
    # TODO сделать описание функции
    if algo not in IMPUTATION_METHODS:
        raise ValueError(f"Unknown mechanism: {algo}")
    
    return IMPUTATION_METHODS[algo](df_incomplete, **kwargs)
    