import os
import datetime
import csv, json
import numpy as np
from collections import OrderedDict
import random

class DataReader(object):
    def __init__(self, dataset_dir, sampling_minute, sampling_size, batch_size, use_json):
        if os.path.exists(dataset_dir / 'train.json') and os.path.exists(dataset_dir / 'valid.json') and use_json:
            self.X_train, self.Y_train = self._load_train_data(dataset_dir)
            self.X_valid, self.Y_valid = self._load_valid_data(dataset_dir)
            if np.shape(self.X_train)[1] != sampling_size:
                self._preprocess_csv_to_json(dataset_dir, sampling_minute, sampling_size)
        else:
            self._preprocess_csv_to_json(dataset_dir, sampling_minute, sampling_size)

        # Define base settings.
        self.batch_size = batch_size
        self.train_data_size = np.shape(self.X_train)[0]
        self.train_batch_num = int(np.ceil(self.train_data_size / self.batch_size))
        self.valid_data_size = np.shape(self.X_valid)[0]
        self.valid_batch_num = int(np.ceil(self.valid_data_size / self.batch_size))
        self.train_batch_index = 0
        self.valid_batch_index = 0
        print("- Train data size =", self.train_data_size)
        print("- Validation data size =", self.valid_data_size)
        assert self.train_data_size == np.shape(self.Y_train)[0]
        assert self.valid_data_size == np.shape(self.Y_valid)[0]

        # Shuffle the train data.
        train_data_index = np.arange(self.train_data_size)
        random.shuffle(train_data_index)
        self.X_train = np.array(self.X_train)[train_data_index]
        self.Y_train = np.array(self.Y_train)[train_data_index]

    def get_train_batch(self):
        X_batch = self.X_train[self.batch_size * self.train_batch_index : self.batch_size * (self.train_batch_index+1)]
        Y_batch = self.Y_train[self.batch_size * self.train_batch_index : self.batch_size * (self.train_batch_index+1)]
        if self.train_batch_index == self.train_batch_num-1:
            self.train_batch_index = 0
        else:
            self.train_batch_index += 1
        return X_batch, Y_batch

    def get_valid_batch(self):
        X_batch = self.X_valid[self.batch_size * self.valid_batch_index: self.batch_size * (self.valid_batch_index + 1)]
        Y_batch = self.Y_valid[self.batch_size * self.valid_batch_index: self.batch_size * (self.valid_batch_index + 1)]
        if self.valid_batch_index == self.valid_batch_num - 1:
            self.valid_batch_index = 0
        else:
            self.valid_batch_index += 1
        return X_batch, Y_batch

    def _load_train_data(self, dataset_dir):
        with open(dataset_dir / 'train.json') as data_file:
            data = json.load(data_file)
            X_train = data['input']
            Y_train = data['output']
        return X_train, Y_train

    def _load_valid_data(self, dataset_dir):
        with open(dataset_dir / 'valid.json') as data_file:
            data = json.load(data_file)
            X_valid = data['input']
            Y_valid = data['output']
        return X_valid, Y_valid

    def _preprocess_csv_to_json(self, dataset_dir, sampling_minute, sampling_size):
        # Use 2013 year data to validation set.
        valid_begin = datetime.datetime(2013, 1, 1, 0, 0)
        train_begin = datetime.datetime(1999, 1, 1, 0, 0) + datetime.timedelta(
            minutes=(np.ceil((sampling_minute * sampling_size) / 180)) * 180)
        train_end = valid_begin - datetime.timedelta(minutes=sampling_minute * sampling_size)
        train_end_date = (train_end - datetime.datetime(2012, 1, 1, 0, 0)).days + 1
        train_end_hour = (train_end - datetime.datetime(2012, 1, 1, 0, 0)).seconds // 3600
        train_end_minute = (train_end - datetime.datetime(2012, 1, 1, 0, 0)).seconds % 3600 / 60
        valid_begin_date = 1
        valid_begin_hour = 0
        valid_begin_minute = 0

        # This is initial input value.(1999-1-0-0)
        minute_past = -1
        np_past = 7.149
        tp_past = 92352.0
        vp_past = 406.0
        bgsm_x_past = -2.174
        bgsm_y_past = -2.598
        bgsm_z_past = 5.55
        bt_past = 6.63

        # Sample and refine the data.
        batch_input = []
        train_batch_input = []
        valid_batch_input = []
        sample_to_train = True
        sample_to_valid = False
        print("Processing solar wind data")
        for year in np.arange(1999, 2014):
            print("- Processing data of", year, "year")
            input_filename = dataset_dir / 'solar_wind' / ('ace_' + str(year) + '.csv')
            with open(input_filename) as csvfile:
                reader = csv.reader(csvfile, delimiter=' ')
                for num, row in enumerate(reader):
                    if num > 14:
                        refined_row = []
                        for elem in row:
                            if elem is not '':
                                elem = float(elem)
                                refined_row.append(elem)
                        missing_minutes= (refined_row[3] - minute_past) % 60 - 1
                        minute_list = np.arange(refined_row[3] - missing_minutes, refined_row[3]+1)
                        # Decide to sample data for train or valid.
                        if year == 2012 and \
                                refined_row[1] == train_end_date and \
                                refined_row[2] == train_end_hour and \
                                train_end_minute in minute_list % 60:
                            sample_to_train = False
                            print("Finish to sample for train set")
                        if year == 2013 and \
                                refined_row[1] == valid_begin_date and \
                                refined_row[2] == valid_begin_hour and \
                                valid_begin_minute in minute_list % 60:
                            sample_to_valid = True
                            print("Start to sample for validation set")
                        # Refining session
                        if 0 in minute_list % sampling_minute:
                            p_density = {np_past != -9999.9: np_past, refined_row[4] != -9999.9: refined_row[4]}.get(True, 0)
                            p_temperature = {tp_past != -9999.9: tp_past, refined_row[5] != -9999.9: refined_row[5]}.get(True, 0)
                            p_speed = {vp_past != -9999.9: vp_past, refined_row[6] != -9999.9: refined_row[6]}.get(True, 0)
                            bgsm_x = {bgsm_x_past != -9999.9: bgsm_x_past,
                                      refined_row[7] != -9999.9: refined_row[7]}.get(True, 0)
                            bgsm_y = {bgsm_y_past != -9999.9: bgsm_y_past,
                                      refined_row[8] != -9999.9: refined_row[8]}.get(True, 0)
                            bgsm_z = {bgsm_z_past != -9999.9: bgsm_z_past,
                                      refined_row[9] != -9999.9: refined_row[9]}.get(True, 0)
                            bt = {bt_past != -9999.9: bt_past, refined_row[10] != -9999.9: refined_row[10]}.get(True, 0)
                            refined_list = [p_density, p_temperature, p_speed, bgsm_x, bgsm_y, bgsm_z, bt]
                            if np.shape(batch_input)[0] < sampling_size:
                                batch_input.append(refined_list)
                            else:
                                batch_input = batch_input[1:sampling_size]
                                batch_input.append(refined_list)
                            if np.shape(batch_input)[0] == sampling_size and \
                                    (refined_row[2]+sampling_minute//60) %3==0 \
                                    and 0 in minute_list % 60:
                                if sample_to_train:
                                    train_batch_input.append(batch_input)
                                elif sample_to_valid:
                                    valid_batch_input.append(batch_input)
                        year_past, date_past, hour_past, minute_past, \
                        np_past, tp_past, vp_past, \
                        bgsm_x_past, bgsm_y_past, bgsm_z_past, \
                        bt_past = refined_row

        print("Processing kp index data...")
        output_filename = dataset_dir / 'kp_index' / 'kp_index.csv'
        train_batch_output = []
        valid_batch_output = []
        with open(output_filename) as csvfile:
            reader = csv.reader(csvfile)
            for num, row in enumerate(reader):
                if num > 0:
                    if int(row[0][:4]) == 2013:
                        for i in np.arange(8):
                            valid_batch_output.append(int(row[i + 1]))
                    else:
                        for i in np.arange(8):
                            time_now = datetime.datetime(int(row[0][:4]), int(row[0][5:7]), int(row[0][8:]), i*3)
                            if time_now < train_begin:
                                pass
                            elif time_now > train_end:
                                pass
                            else:
                                train_batch_output.append(int(row[i + 1]))
        # Save to json file
        train_data = OrderedDict()
        valid_data = OrderedDict()
        train_data["name"] = 'train_dataset'
        train_data["input"] = train_batch_input
        train_data["output"] = train_batch_output
        valid_data["name"] = 'valid_dataset'
        valid_data["input"] = valid_batch_input
        valid_data["output"] = valid_batch_output
        self.X_train = train_batch_input
        self.Y_train = train_batch_output
        self.X_valid = valid_batch_input
        self.Y_valid = valid_batch_output

        print("Saving train dataset to json file")
        with open(dataset_dir / 'train.json', 'w', encoding="utf-8") as make_file:
            json.dump(train_data, make_file, ensure_ascii=False, indent="\t")
        print("Saving validation dataset to json file")
        with open(dataset_dir / 'valid.json', 'w', encoding="utf-8") as make_file:
            json.dump(valid_data, make_file, ensure_ascii=False, indent="\t")

