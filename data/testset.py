import os
import datetime
import csv, json
import numpy as np
from collections import OrderedDict
import random

class TestReader(object):
    def __init__(self, dataset_dir, sampling_minute, sampling_size, use_json):
        if os.path.exists(dataset_dir / 'test_input.json') and use_json:
            self.X_test = self._load_test_data(dataset_dir)
        else:
            self._preprocess_csv_to_json(dataset_dir, sampling_minute, sampling_size)

    def _load_test_data(self, dataset_dir):
        with open(dataset_dir / 'test_input.json') as data_file:
            data = json.load(data_file)
            X_test = data['input']
        return X_test

    def _preprocess_csv_to_json(self, dataset_dir, sampling_minute, sampling_size):
        batch_input = []
        test_batch_input = []
        start_sampling = False
        input_filename = dataset_dir / 'test' / 'input' / 'preprocessed_solar_wind.csv'
        # This is initial input value.(?-180-0-1)
        minute_past = -1
        np_past = 2.416
        tp_past = 65701
        vp_past = 402.18
        bgsm_x_past = -4.212
        bgsm_y_past = -3.982
        bgsm_z_past = 2.445
        bt_past = 6.298
        with open(input_filename) as csvfile:
            reader = csv.reader(csvfile)
            for num, row_str in enumerate(reader):
                if num == 0:
                    pass
                else:
                    row = []
                    for r in row_str:
                        row.append(float(r))
                    missing_minutes = (row[2] - minute_past) % 60 - 1
                    minute_list = np.arange(row[2] - missing_minutes, row[2] + 1)
                    # Sample to test batch.
                    if row[0] == 1 and row[1] == 0 and row[2] ==0:
                        print("Start to sample for test set")
                        start_sampling = True
                    if 0 in minute_list % sampling_minute:
                        p_density = {np_past != -9999.9: np_past, row[3] != -9999.9: row[3]}.get(True,0)
                        p_temperature = {tp_past != -9999.9: tp_past, row[4] != -9999.9: row[4]}.get(True, 0)
                        p_speed = {vp_past != -9999.9: vp_past, row[5] != -9999.9: row[5]}.get(True, 0)
                        bgsm_x = {bgsm_x_past != -9999.9: bgsm_x_past,
                                  row[6] != -9999.9: row[6]}.get(True, 0)
                        bgsm_y = {bgsm_y_past != -9999.9: bgsm_y_past,
                                  row[7] != -9999.9: row[7]}.get(True, 0)
                        bgsm_z = {bgsm_z_past != -9999.9: bgsm_z_past,
                                  row[8] != -9999.9: row[8]}.get(True, 0)
                        bt = {bt_past != -9999.9: bt_past, row[9] != -9999.9: row[9]}.get(True, 0)
                        refined_list = [p_density, p_temperature, p_speed, bgsm_x, bgsm_y, bgsm_z, bt]
                        if np.shape(batch_input)[0] < sampling_size:
                            batch_input.append(refined_list)
                        else:
                            batch_input = batch_input[1:sampling_size]
                            batch_input.append(refined_list)
                        if start_sampling and row[1] % 3 == 0 and 0 in minute_list % 60:
                            test_batch_input.append(batch_input)
                    date_past, hour_past, minute_past, \
                    np_past, tp_past, vp_past, \
                    bgsm_x_past, bgsm_y_past, bgsm_z_past, bt_past = row

        self.X_test = test_batch_input
        # Save to json file
        test_data = OrderedDict()
        test_data["name"] = 'test_dataset'
        test_data["input"] = test_batch_input
        print("Saving test dataset to json file")
        with open(dataset_dir / 'test_input.json', 'w', encoding="utf-8") as make_file:
            json.dump(test_data, make_file, ensure_ascii=False, indent="\t")
