import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import ppscore as pps
import time
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from yellowbrick.regressor import PredictionError

def regression_visualization(model, X_train, X_test, y_train, y_test):
    visualizer = PredictionError(model)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    plt.title('Score visualization')
    plt.legend()
    st.pyplot()

def define_model():
    model_select = st.selectbox('Select Model', ('None', 'LinearRegression',
                                                         'RandomForest', 'SVM',
                                                         'GradientBoosting', 'KNN'))
    if model_select == 'None':
        return None
    else:
        if model_select == 'LinearRegression':
            model = LinearRegression()
            return model
        if model_select == 'RandomForest':
            st.markdown('**Parameters Tuning**')
            bootstrap = st.selectbox('Bootstrap', (True, False))
            max_features = st.selectbox('Max Features', ('auto', 'sqrt'))
            max_depth_slider = st.slider('Max Depth', 10, 100)
            min_samples_leaf_slider = st.slider('Min Samples Leaf', 1, 5)
            min_samples_split_slider = st.slider('Min Samples Split', 2, 10)
            n_estimators_slider = st.slider('Number of Estimators', 200, 2000)
            model = RandomForestRegressor(bootstrap=bootstrap, max_depth=max_depth_slider,
                                          max_features=max_features,
                                          min_samples_leaf=min_samples_leaf_slider,
                                          min_samples_split=min_samples_split_slider,
                                          n_estimators=n_estimators_slider)
            return model

        if model_select == 'KNN':
            st.markdown('**Parameters Tuning**')
            k = st.slider('Number of Neighbors', 1, 20)
            model = KNeighborsRegressor(n_neighbors=k)
            return model

        if model_select == 'SVM':
            st.markdown('**Parameters Tuning**')
            degree_slider = st.slider('Degree', 2, 6)
            model = SVR(degree=degree_slider)
            return model

        if model_select == 'GradientBoosting':
            st.markdown('**Parameters Tuning**')
            learning_rate = st.slider('Learning Rate', 0.005, 0.1)
            min_samples_leaf_slider2 = st.slider('Min Samples Leaf', 1, 5)
            min_samples_split_slider2 = st.slider('Min Samples Split', 2, 10)
            max_depth_slider2 = st.slider('Max Depth', 10, 100)
            n_estimators_slider2 = st.slider('Number of Estimators', 200, 2000)
            model = GradientBoostingRegressor(learning_rate=learning_rate,
                                              min_samples_leaf=min_samples_leaf_slider2,
                                              min_samples_split=min_samples_split_slider2,
                                              max_depth=max_depth_slider2,
                                              n_estimators=n_estimators_slider2)
            return model


def model_score(X, y, model):
    kfold = KFold(n_splits=5, shuffle=True)
    score = -np.mean(cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error'))
    return st.markdown('Train RMSE: {}'.format(np.sqrt(score)))

def split_data(df):
    target = st.selectbox('Define Target', df.columns[::-1])
    X = df.drop(target, axis=1)
    y = df[target]
    split_slider = st.slider('Test size', 0.1, 0.8)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_slider)
    return X_train, X_test, y_train, y_test

def feature_score(df):
    score_check = st.sidebar.checkbox('View PPScore between features ')
    if score_check:
        score_var1 = st.sidebar.selectbox('Select feature', df.columns)
        score_var2 = st.sidebar.selectbox('Select target', df.columns)
        st.markdown('**Prediction of {} using {}**'.format(score_var2, score_var1))
        score = pps.score(df, score_var1, score_var2)
        st.write(score['ppscore'])

def corr_matrix_heatmap(df):
    corr_method = st.sidebar.selectbox('Choose correlation heatmap',
                                       ('None', 'Pearson', 'Spearman'))
    if corr_method == 'None':
        return
    elif corr_method == 'Pearson':
        corr = df.corr(method='pearson')
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr)
        st.pyplot()
    elif corr_method == 'Spearman':
        corr = df.corr(method='spearman')
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr)
        st.pyplot()


def normalize_df(df, numeric_data):
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer
    norm_method = st.sidebar.selectbox('Choose normalization method',
                                       ('None', 'StandardScaler', 'MinMaxScaler', 'RobustScaler',
                                        'QuantileTransformer', 'PowerTransformer'))
    if norm_method == 'None':
        return df[numeric_data]
    else:
        if norm_method == 'StandardScaler':
            scaler = StandardScaler()
        elif norm_method == 'MinMaxScaler':
            scaler = MinMaxScaler()
        elif norm_method == 'RobustScaler':
            scaler = RobustScaler()
        elif norm_method == 'QuantileTransformer':
            scaler = QuantileTransformer()
        elif norm_method == 'PowerTransformer':
            scaler = PowerTransformer()

        df_scaled = scaler.fit_transform(df[numeric_data])
        st.success('Done!')
        return df_scaled

def feature_importances(df):
    pass


def show_missing_values(null_data, null_data_all, df):
    st.subheader('**Missing Values**')
    if null_data_all.sum() == 0:
        st.markdown('No missing values')
    else:
        for index, value in null_data.sort_values(ascending=False).items():
            perc_nulls = np.round(value / len(df.index) * 100, 2)
            st.write(index, ' ---> Missing Values by total: ', value, 'of ', len(df.index))
            st.write('Percentage per Missing Value: ', perc_nulls, '%\n')
        plt.figure(figsize=(10, 8))
        sns.barplot(null_data.sort_values(ascending=False),
                    null_data.sort_values(ascending=False).index)
        st.pyplot()

def make_histogram(var, df):
    st.write("**Mean:**", df[var].mean())
    st.write("**Std:**", df[var].std())
    hist = sns.distplot(df[var], bins=60)
    plt.ylabel('Frequency')
    return hist

def make_scatter(var1, var2, df):
    data = df[[var1, var2]]
    st.write('**Correlation**', data.corr(method='pearson'))
    scatter = plt.scatter(df[var1], df[var2], alpha=0.8)
    return scatter

def make_barplot(num_var, cat_var, df):
    bar = sns.barplot(num_var, cat_var, data=df, palette='prism')
    return bar


def main():

### Df infos
    st.title('Streamlit Regression Deploy Pipeline')
    select_dataset = st.sidebar.selectbox('Select your Dataset', ('None', 'House Prices'))
    if select_dataset == 'None':
        st.markdown('**No Dataset selected**')
    else:
        if select_dataset == 'House Prices':
            st.subheader('House Prices Dataset')
            df = pd.read_csv('house_prices_train.csv')
            df.set_index('Id', inplace=True)
            target = 'SalePrice'
        slider = st.slider('Header size', 1, 50)
        st.dataframe(df.head(slider))
        st.write('Data Shape: ', df.shape)
        st.subheader('**Statistical Describe of Data:**')
        st.dataframe(df.describe())
        aux = pd.DataFrame(index=df.columns,
                           data={'type': df.dtypes, 'uniques': df.nunique(), 'nulls': df.isnull().sum()})
        numeric_vars = list(aux[aux['type'] != 'object'].index)
        object_vars = list(aux[aux['type'] == 'object'].index)
        st.subheader('**DataFrame with features characteristics**')
        st.dataframe(aux)

###     Missing values
        missing_values_button = st.sidebar.button('Show Missing Values')
        null_data_all = df.isnull().sum()
        null_data = null_data_all.drop(null_data_all[null_data_all == 0].index)
        if missing_values_button:
            show_missing_values(null_data, null_data_all, df)

        handle_missing = st.sidebar.selectbox('Handle Missing Values?', ('No', 'Yes'))
        if handle_missing != 'No':

            drop_null = st.sidebar.checkbox('Drop data with more than 70% null values?')
            if drop_null:
                for index, value in null_data.items():
                    perc_nulls = np.round(value / len(df.index) * 100, 2)
                    if perc_nulls >= 70:
                        df.drop(index, axis=1, inplace=True)
                with st.spinner('Wait for it...'):
                    time.sleep(0.3)
                st.success('Done!')
                null_data_all = df.isnull().sum()
                null_data = null_data_all.drop(null_data_all[null_data_all == 0].index)
                missing_values_button2 = st.sidebar.button('Show Missing Values After Drop')
                if missing_values_button2:
                    show_missing_values(null_data, null_data_all, df)

            aux = pd.DataFrame(index=df.columns,
                               data={'type': df.dtypes, 'uniques': df.nunique(), 'nulls': df.isnull().sum()})
            numeric_vars = list(aux[aux['type'] != 'object'].index)
            object_vars = list(aux[aux['type'] == 'object'].index)
            columns = list(df.columns)

            fill_numeric_nulls = st.sidebar.radio('Choose how to handle missing numeric values:',
                                                  ('Do Nothing', 'Fill with 0', 'Fill with mean',
                                                   'Drop columns with missing values'))
            if fill_numeric_nulls != 'Do Nothing':
                for col in numeric_vars:
                    if fill_numeric_nulls == 'Fill with 0':
                        df[col].fillna(0, inplace=True)
                    elif fill_numeric_nulls == 'Fill with mean':
                        df[col].fillna(df[col].mean(), inplace=True)
                    elif fill_numeric_nulls == 'Drop columns with missing values':
                        if col in null_data.index:
                            df.drop(col, axis=1, inplace=True)

                null_data_all = df.isnull().sum()
                null_data = null_data_all.drop(null_data_all[null_data_all == 0].index)
                missing_values_button3 = st.sidebar.button('Show Missing Values After Filling Numerical Variables', )
                if missing_values_button3:
                    show_missing_values(null_data, null_data_all, df)

            fill_categorical_nulls = st.sidebar.radio('Choose how to handle missing categorical values:',
                                                  ('Do Nothing','Fill with "None"',
                                                   'Drop columns with missing values'))
            if fill_categorical_nulls != 'Do Nothing':
                for col in object_vars:
                    if fill_categorical_nulls == 'Fill with "None"':
                        df[col].fillna('None', inplace=True)
                    elif fill_categorical_nulls == 'Drop columns with missing values':
                        if col in null_data.index:
                            df.drop(col, axis=1, inplace=True)

                null_data_all = df.isnull().sum()
                null_data = null_data_all.drop(null_data_all[null_data_all == 0].index)
                missing_values_button4 = st.sidebar.button('Show Missing Values After Filling Categorical Variables')
                if missing_values_button4:
                    show_missing_values(null_data, null_data_all, df)

            aux = pd.DataFrame(index=df.columns,
                               data={'type': df.dtypes, 'uniques': df.nunique(), 'nulls': df.isnull().sum()})
            numeric_vars = list(aux[aux['type'] != 'object'].index)
            object_vars = list(aux[aux['type'] == 'object'].index)
            columns = list(df.columns)
            df_after_null = st.sidebar.button('Show df after cleaning')
            if df_after_null:
                st.dataframe(df.head(10))
                st.write('Data Shape: ', df.shape)


###     Visualization
        sidebar_visualization = st.sidebar.selectbox('Visualization of variable',
                                                     ('None', 'Histogram', 'Scatter', 'Bar'))
        if sidebar_visualization == 'None':
            pass
        else:
            if sidebar_visualization == 'Scatter':
                scatter_var1 = st.sidebar.selectbox('Select variable for x axis:', numeric_vars)
                scatter_var2 = st.sidebar.selectbox('Select variable for y axis:', numeric_vars)
                make_scatter(scatter_var1, scatter_var2, df)
                st.pyplot()

            if sidebar_visualization == 'Bar':
                bar_num_var = st.sidebar.selectbox('Select numeric Variable:', numeric_vars)
                bar_cat_var = st.sidebar.selectbox('Select categorical Variable:', object_vars)
                make_barplot(bar_cat_var, bar_num_var, df)
                st.pyplot()

            if sidebar_visualization == 'Histogram':
                col_hist = st.sidebar.selectbox('Select column:', numeric_vars)
                make_histogram(col_hist, df)
                st.pyplot()
                normalize_var = st.sidebar.button('Log Normalize Feature')
                if normalize_var:
                    df[col_hist] = np.log1p(df[col_hist])
                    make_histogram(col_hist, df)
                    st.pyplot()

###     Scaling Data
        df[numeric_vars] = normalize_df(df, numeric_vars)
        show_df_button = st.sidebar.button('View numeric features')
        if show_df_button:
            st.subheader('Numeric features after Normalization')
            st.dataframe(df[numeric_vars])

###     Transform Categorical Vars and drop uniques
        dummies_check = st.sidebar.checkbox('Get Dummies')
        if dummies_check:
            df[object_vars] = df[object_vars].astype('category')
            df_dummies = pd.get_dummies(df, drop_first=True)
        else:
            df_dummies = df

###     Correlations
        corr_matrix_heatmap(df)
        feature_score(df)

###     Split Data
        st.subheader('**Start Regression Here**')
        X_train, X_test, y_train, y_test = split_data(df_dummies)

###     Model Selection and Evaluation
        model = define_model()
        if df_dummies.isnull().sum().any():
            st.markdown('**Handle missing values first**')
            model = None
        elif dummies_check == False:
            st.markdown('**Handle categorical features first (Get dummies)**')
        elif model == None:
            st.markdown(' ')
        else:
            model_score(X_train, y_train, model)
            regression_visualization(model, X_train, X_test, y_train, y_test)


main()