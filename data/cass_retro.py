import os
import math
import logging
import numpy as np

import pandas as pd
from tensorflow.python.data import AUTOTUNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
import tensorflow as tf
from tensorflow_federated.python.simulation.datasets import ClientData

from data.abstract_dataset import AbstractDataset
from data.feature_selector import FeatureSelector

DATA_PATH = os.path.abspath("~/workspace/federated-complication-prediction/data/pre_with_organs.csv")
DATA_INFOS_PATH = os.path.abspath("~/workspace/federated-complication-prediction/src/data/cass_retro/cass_retro_data_infos.csv")
PREDICTION_TARGET = 'target_death_within_primary_stay'
NORMALISE = 'minmax'

CID_RENAMES = {
    'meta_system_0': 'Oesophagus',
    'meta_system_1': 'Stomach',
    'meta_system_2': 'Colorectum',
    'meta_system_3': 'Liver',
    'meta_system_4': 'Pancreas'
}

CIDS = {
    'Oesophagus': 0,
    'Stomach': 1,
    'Colorectum': 2,
    'Liver': 3,
    'Pancreas': 4
}

class CassRetroDataset(AbstractDataset):
    @property
    def class_labels(self):
        return ['Survived', 'Died']

    @property
    def dataset_size(self):
        # todo: get the actual dataset size
        return {'train': 5232, 'test': 656}

    @property
    def avg_local_dataset_size(self):
        # todo: get the actual dataset size
        return 436, 0

    def get_default_accuracy(self):
        # todo: get the actual default accuracy
        return 0.748

    def get_dataset_size_for_client(self, client_id):
        # todo: get the actual dataset size for client
        return 436

    def _load_tff_dataset(self):
        feature_set = ['pre']
        complete_data = pd.read_csv(DATA_PATH, delimiter=',')
        data_infos = pd.read_csv(DATA_INFOS_PATH)
        imputer = KNNImputer(n_neighbors=5, weights='uniform')

        def get_all_features(include_drop_columns=True):
            return list(data_infos.loc[~data_infos['endpoint'] & (include_drop_columns | ~data_infos['drop']), 'column_name'])

        def get_binary_features(include_drop_columns=True):
            return list(data_infos.loc[~data_infos['endpoint'] &
                                            (include_drop_columns | ~data_infos['drop']) &
                                            (data_infos['data_type'] == 'B'), 'column_name'])

        def get_categorical_features(include_drop_columns=True):
            return list(data_infos.loc[~data_infos['endpoint'] &
                                            (include_drop_columns | ~data_infos['drop']) &
                                            (data_infos['data_type'] == 'C'), 'column_name'])

        def get_numerical_features(include_drop_columns=True):
            return list(data_infos.loc[~data_infos['endpoint'] &
                                            (include_drop_columns | ~data_infos['drop']) &
                                            (data_infos['data_type'] == 'N'), 'column_name'])

        def get_all_endpoints(include_drop_columns=True):
            return list(data_infos.loc[data_infos['endpoint'] & (include_drop_columns | ~data_infos['drop']), 'column_name'])

        # Assert the length of the intersection of data and data infos
        assert len(
            set(complete_data.columns).intersection(
                set(get_all_features() + get_all_endpoints()))) == len(
            set(complete_data.columns)), "Column set doesn't match: " + \
                                         str([col for col in complete_data.columns if
                                              col not in set(complete_data.columns).intersection(
                                                  set(get_all_features() + get_all_endpoints()))])

        drop_columns_list = []
        if feature_set is not None:
            drop_columns_list.extend(list(data_infos.loc[
                                              ~data_infos['endpoint'] & ~data_infos['input_time'].isin(
                                                  feature_set), 'column_name']))

        # Drop features
        drop_columns_list.extend(list(data_infos.loc[data_infos['drop'], 'column_name']))
        # Remove the updated values from esophagus_info_updated
        logging.debug(
            list(set(data_infos.column_name.values).difference(set(complete_data.columns.values))))
        difference = list(set(data_infos.column_name.values).difference(set(complete_data.columns.values)))
        data_infos = data_infos[~data_infos["column_name"].isin(difference)]
        complete_data.drop(columns=drop_columns_list, inplace=True, errors='ignore')

        # don't drop missing value rows
        # if drop_missing_value > 0:
        #     # Calculate the minimum amount of columns that have to contain a value
        #     min_count = int(((100 - drop_missing_value) / 100) * complete_data.shape[1] + 1)
        #
        #     # Drop rows not meeting threshold
        #     complete_data = complete_data.dropna(axis=0, thresh=min_count)

        # Extract features and endpoints
        X = complete_data[data_infos.loc[
            ~(data_infos['endpoint'] | data_infos['column_name'].isin(drop_columns_list)), 'column_name']]
        Y = complete_data[data_infos.loc[
            data_infos['endpoint'] & ~data_infos['column_name'].isin(drop_columns_list), 'column_name']]

        # === Dataset-specific preprocessing steps ===
        # all_error_idxs = X[X['height'] <= X['weight']].index
        all_error_idxs = X[
            ~(X['height'].between(90, 220) & X['weight'].between(30, 250)) & X['height'].notna() &
            X['weight'].notna()].index

        # Swap height and weight if they look like they are swapped
        w_h_swapped_idx = X[X['height'].between(30, 150) &
                            X['weight'].between(120, 250) &
                            (X['height'] < X['weight'])].index
        X.loc[w_h_swapped_idx, ['height', 'weight']] = X.loc[w_h_swapped_idx, ['weight', 'height']].values
        logging.info(f'Swapped {len(w_h_swapped_idx)} heights and weights.')

        # Divide weight by 10 if it makes sense
        weight_decimal_error_idx = X[X['height'].between(120, 220) &
                                     (X['weight'] / 10).between(30.0, 250.0)].index
        X.loc[weight_decimal_error_idx, 'weight'] = X.loc[weight_decimal_error_idx, 'weight'].values / 10
        logging.info(f'Fixed {len(weight_decimal_error_idx)} weights decimal errors.')

        # Multiply height by 10 if it makes sense
        height_decimal_error_idx = X[
            X['weight'].between(30, 250) & (X['height'] * 10).between(120.0, 220.0)].index
        X.loc[height_decimal_error_idx, 'height'] = X.loc[height_decimal_error_idx, 'height'].values * 10
        logging.info(f'Fixed {len(height_decimal_error_idx)} height decimal errors.')

        both_height_weight_zero_idxs = X[(X['height'] == 0) & (X['weight'] == 0)].index
        X.loc[both_height_weight_zero_idxs, ['height', 'weight']] = np.nan, np.nan
        logging.info(f'Replaced {len(both_height_weight_zero_idxs)} double zeros in height/weight with nan.')

        equal_height_weight_idxs = X[(X['weight'] == X['height']) & (X['weight'] != 0)].index
        equal_height_weight_idxs = equal_height_weight_idxs.intersection(all_error_idxs)
        replacements = []
        for idx in equal_height_weight_idxs:
            if X.loc[idx, 'height'] < 100:
                replacements.append(True)
                X.loc[idx, 'height'] = np.nan
            else:
                replacements.append(False)
        equal_height_weight_idxs = equal_height_weight_idxs[replacements]
        logging.info(f'Replaced {sum(replacements)} heights with nan, where height == weight and height < 100')

        height_zero_idxs = X[X['height'] == 0].index
        X.loc[height_zero_idxs, 'height'] = np.nan
        logging.info(f'Replaced {len(height_zero_idxs)} heights = 0 with nan')

        weight_zero_idxs = X[X['weight'] == 0].index
        X.loc[weight_zero_idxs, 'weight'] = np.nan
        logging.info(f'Replaced {len(weight_zero_idxs)} weights = 0 with nan')

        remaining_error_idxs = (all_error_idxs.difference(w_h_swapped_idx)
                                .difference(weight_decimal_error_idx)
                                .difference(height_decimal_error_idx)
                                .difference(both_height_weight_zero_idxs)
                                .difference(equal_height_weight_idxs)
                                .difference(height_zero_idxs)
                                .difference(weight_zero_idxs)
                                )

        X.loc[remaining_error_idxs, ['height', 'weight']] = np.nan, np.nan
        logging.info(f'Set remaining {len(remaining_error_idxs)} records with weight height/weight to nan.')

        # Drop age < 18
        underaged_idxs = X[X['age'] < 18].index
        X.drop(index=underaged_idxs, inplace=True)
        Y.drop(index=underaged_idxs, inplace=True)
        logging.info(f'Dropped {len(underaged_idxs)} underaged records.')

        # Drop weight < 30
        too_low_weight_idxs = X[X['weight'] < 30.0].index
        X.drop(index=too_low_weight_idxs, inplace=True)
        Y.drop(index=too_low_weight_idxs, inplace=True)
        logging.info(f'Dropped {len(too_low_weight_idxs)} records with weight < 30.')

        # Drop height > 220
        too_large_height_idxs = X[X['height'] > 220.0].index
        X.drop(index=too_large_height_idxs, inplace=True)
        Y.drop(index=too_large_height_idxs, inplace=True)
        logging.info(f'Dropped {len(too_large_height_idxs)} records with height > 220')

        # Drop missing meta_ops
        if 'surgery_ops_0' in X.columns:
            missing_ops_idxs = X[X['surgery_ops_0'].isna()].index
            X.drop(index=missing_ops_idxs, inplace=True)
            Y.drop(index=missing_ops_idxs, inplace=True)
            logging.info(f'Dropped {len(missing_ops_idxs)} records missing OPS code')

        X.reset_index(drop=True, inplace=True)
        Y.reset_index(drop=True, inplace=True)

        y = Y[PREDICTION_TARGET]
        X = X[y.notna()]
        y = y[y.notna()].astype(int)

        # preprocess.py
        fs_operations = ['single_unique']

        # Apply FeatureSelector functionality
        fs = FeatureSelector()
        if 'single_unique' in fs_operations:
            fs.identify_single_unique(X)
            logging.debug(fs.record_single_unique)
        if 'missing' in fs_operations:
            fs.identify_missing(X, missing_threshold=0.5)
            logging.debug(fs.record_missing)
        if 'collinear' in fs_operations:
            fs.identify_collinear(X, correlation_threshold=0.95)
            logging.debug(fs.record_collinear)
            # logging.info(f"Removing colinear columns: {fs.removal_ops['collinear']}")
        X = fs.remove(X, fs_operations, one_hot=False)

        # Fix strings in Binary columns
        for binary_col in get_binary_features():
            if binary_col in X.columns:
                unique_vals = list(np.unique(X[binary_col].values))
                assert len(unique_vals) <= 2, f'Binary column {binary_col} has more than 2 unique values: {unique_vals}'
                if len(unique_vals) != 1 and unique_vals != [0, 1]:
                    logging.info(f'Renaming entries from {binary_col}: {unique_vals[0]} -> 0; {unique_vals[1]} -> 1')
                    X[binary_col].replace({unique_vals[0]: 0,
                                           unique_vals[1]: 1}, inplace=True)

        categorical_features = [col for col in X.columns if col in get_categorical_features()]
        # X[categorical_features] = X[categorical_features].fillna(value=0)

        # One-hot-encode categorical features
        X = pd.get_dummies(X, columns=categorical_features, dummy_na=False)

        X_numerical = X[[col for col in X.columns if col in get_numerical_features()]]
        X_binary = X.drop(columns=[col for col in X.columns if col in get_numerical_features()])

        X_numerical_feature_names = X_numerical.columns
        # Interpolate numerical features
        if imputer is not None:
            logging.debug('Running Imputer...')
            X_numerical = imputer.fit_transform(X_numerical)
            X_numerical = pd.DataFrame(X_numerical, columns=X_numerical_feature_names)

            # recalculate BMI
            if 'bmi' in X_numerical.columns:
                X_numerical['bmi'] = round(
                    X_numerical['pre_op_weight_in_kg'] / (X_numerical['height_in_cm'] / 100) ** 2, 1)
                logging.debug(f'New BMI statistics:\n{X_numerical["bmi"].describe()}')
        else:
            X_numerical = X_numerical.dropna(axis=1)
            X_numerical_feature_names = X_numerical.columns

        if NORMALISE is not None:
            logging.debug(f'Normalising numerical features using {NORMALISE} scaling...')
            if NORMALISE == 'standard':
                # Normalise numerical features
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()
            X_numerical = scaler.fit_transform(X_numerical)
            X_numerical = pd.DataFrame(X_numerical, columns=X_numerical_feature_names)

        # Build X and y arrays
        X = pd.concat([X_binary, X_numerical], axis=1)

        logging.info(f'Final Dataset dimensions = {X.shape}')

        # Split dataset for TFF
        split_clients_by = 'meta_system'

        def create_tf_dataset_for_client_fn(client_id):
            if split_clients_by in X.columns:
                client_X = X[X[split_clients_by] == CIDS[client_id]]
                client_y = y[X[split_clients_by] == CIDS[client_id]]
            else:
                client_X = X[X[f'{split_clients_by}_{CIDS[client_id]}'] == 1]
                client_y = y[X[f'{split_clients_by}_{CIDS[client_id]}'] == 1]

            # drop columns
            client_X = client_X.drop(columns=client_X.columns[client_X.columns.str.startswith(split_clients_by)])

            return tf.data.Dataset.from_tensor_slices((client_X, client_y))

        return ClientData.from_clients_and_fn(list(CIDS.keys()),
                                              create_tf_dataset_for_client_fn)

    def _load_tf_dataset(self):
        pass
        # # Old stuff
        # complete_train_ds = (train_data
        #                      .shuffle(buffer_size=self.dataset_size['train'], reshuffle_each_iteration=False)
        #                      .map(read_img, num_parallel_calls=AUTOTUNE))
        # self.train_ds = (complete_train_ds
        #                  .take(math.floor(self.dataset_size['train'] * (1 - self.cfg.val_fraction)))
        #                  .cache())
        # self.val_ds = (complete_train_ds
        #                .skip(math.floor(self.dataset_size['train'] * (1 - self.cfg.val_fraction)))
        #                .cache())
        # self.test_ds = (test_data
        #                 .shuffle(buffer_size=self.dataset_size['test'], reshuffle_each_iteration=False)
        #                 .map(read_img, num_parallel_calls=AUTOTUNE)
        #                 .cache())
