from collections import defaultdict
import os
from math import nan
import numpy as np
from pathlib import Path
import pickle
import progressbar
import re
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple
import uuid

PBAR_DATA_DIR = os.getenv("HOME") + '/.config/pbar'
PBAR_LOG_DIR = PBAR_DATA_DIR + '/logs'
# FIXME: We should regen a temp file each time
PBAR_TEMP_FILE = PBAR_DATA_DIR + '/pbar_tmp.out'

MAX_LOG_GRANULARITY = 200


class SystemCall(object):
    def __init__(self, epoch_time: float, call_name: str, arguments: List[int], result: int):
        self.epoch_time = epoch_time
        self.call_name = call_name
        self.arguments = arguments
        self.result = result

    @staticmethod
    def parse(s: str):
        if len(s) < 1 or 'unfinished' in s or 'resumed' in s or '=' not in s or '(' not in s or ')' not in s or '.' not in s:
            return None

        try:
            _pid, epoch_time, call = re.split('\\s\\s\\s|\\s\\s|\\s', s, maxsplit=2)

            if call.startswith('+') or call.startswith('-'):
                return None

            try:
                epoch_time = float(epoch_time)
            except ValueError:
                return None

            name, rest = call.split('(', maxsplit=1)
            args_string, result_string = rest.split('=', maxsplit=1)
            if not result_string.strip():
                return None

            args_string = args_string.strip().split(')')[0]
            result_string = result_string.split('(')[0].split('E')[0]

            name = name.strip()
            if len(args_string) == 0:
                args = []
            else:
                args = [int(v, 0) for v in args_string.strip().split(',')]
            if result_string.strip() == '?':
                result_string = '0'
            elif result_string.strip() == '0x':
                return None
            result = int(result_string, 0)

            return SystemCall(epoch_time, name, args, result)
        except:
            print("failed to parse", repr(s))
            raise

    def __repr__(self):
        return 'time={}|call={}|args={}|res={}'.format(self.epoch_time, self.call_name, self.arguments, self.result)


class ProgramLog(object):
    def __init__(self, cmd: List[str], calls: List[SystemCall]):
        if len(calls) == 0:
            raise ValueError('Cannot have a log of no calls!')

        self.cmd = cmd
        self.calls = calls
        self.execution_duration = calls[-1].epoch_time - calls[0].epoch_time

    # This method maps a log into a feature map (some features may be missing for some logs)
    def to_feature_map(self) -> Dict[str, Any]:
        # FIXME: Command counts are not good enough on their own
        feature_dict = defaultdict(float)
        for call in self.calls:
            feature_dict[call.call_name + '--count'] += 1.

        feature_dict['current_execution_duration'] = self.duration()

        # for i, arg in enumerate(self.cmd[:10]):
        #     feature_dict['arg_' + str(i + 1)] = arg

        return feature_dict

    def duration(self):
        return self.execution_duration


def read_program_logs() -> List[ProgramLog]:
    pathlist = Path(PBAR_LOG_DIR).glob('*.pkl')
    logs = []
    for path in pathlist:
        with open(str(path), 'rb') as pkl:
            logs.append(pickle.load(pkl))
    return logs


def write_program_log(p: ProgramLog):
    os.makedirs(PBAR_LOG_DIR, exist_ok=True)

    path = PBAR_LOG_DIR + '/program_log_' + str(uuid.uuid4()) + '.pkl'
    with open(path, 'wb') as log_file:
        pickle.dump(p, log_file)


class Model(object):
    def __init__(self):
        self.features = []
        # self.model = KNeighborsRegressor(n_neighbors=3, p=2)
        # self.model = LinearRegression()
        self.model = RandomForestRegressor(n_estimators=150)
        # self.model = AdaBoostRegressor(n_estimators=200)
        self.imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
        self.scaler = StandardScaler()
        self.trained = False

    @staticmethod
    def get_labeled_logs(dataset: List[ProgramLog]) -> List[Tuple[ProgramLog, float]]:
        # Logic here is a bit messy -- basically just want MAX_LOG_GRANULARITY entries at most for one log
        items = []
        for log in dataset:
            total_time = log.duration()
            calls_per_entry = max(len(log.calls) // MAX_LOG_GRANULARITY, 1)

            acc = []
            for syscall in log.calls:
                acc.append(syscall)
                if len(acc) > 1 and len(acc) % calls_per_entry == 0:
                    new_log = ProgramLog(log.cmd, acc)
                    items.append((new_log, (new_log.duration() / total_time) * 1))
        print("labeled log count", len(items))
        return items

    def update_features(self, logs: List[ProgramLog]) -> ():
        features = set(self.features)
        for log in logs:
            features.update(log.to_feature_map().keys())
        self.features = list(sorted(features))

    def extract_features(self, log: ProgramLog) -> List[Any]:
        vec = []
        feature_map = log.to_feature_map()
        for feature in self.features:
            vec.append(feature_map.get(feature, nan))
        return vec

    def train(self, cmd: List[str], dataset: List[ProgramLog]):
        print("Generating trimmed dataset...")
        # Create a trimmed dataset of commands that prefix match -- getting as specific as possible
        trimmed_dataset = dataset
        i = 0
        while len(cmd) > i:
            candidate = list(filter(lambda log: i < len(log.cmd) and log.cmd[i] == cmd[i], dataset))
            if len(candidate) == 0:
                break
            else:
                trimmed_dataset = candidate
                i += 1

        print("Generating labeled logs...")
        trimmed_dataset = Model.get_labeled_logs(trimmed_dataset)
        self.update_features([log for log, label in trimmed_dataset])

        if len(trimmed_dataset) > 0:
            x = []
            y = []
            print("Extracting features from labeled logs...")
            for log, label in trimmed_dataset:
                x.append(self.extract_features(log))
                y.append(label)

            x = np.array(x)
            y = np.array(y)

            print("Preprocessing data...")
            self.imp.fit(x, y)
            x = self.imp.transform(x)
            self.scaler.fit(x, y)
            x = self.scaler.transform(x)

            print("Fitting model...")
            self.model.fit(x, y)
            print("model training accuracy = ", self.model.score(x, y) * 100, "%")
            self.trained = True

    def predict_completion(self, log: ProgramLog) -> float:
        if not self.trained:
            return 1.
        features = [self.extract_features(log)]
        features = self.imp.transform(features)
        features = self.scaler.transform(features)

        return self.model.predict(features)

    def check_accuracy(self, labled_logs: List[Tuple[ProgramLog, float]]):
        x = [self.extract_features(log) for log, _ in labled_logs]
        x = self.imp.transform(x)
        x = self.scaler.transform(x)

        y = [label for _, label in labled_logs]
        return self.model.score(x, y)



def line_generator(f):
    buf = ""

    while True:
        where = f.tell()

        byte = os.read(f.fileno(), 1)
        while byte:
            buf += byte.decode("utf-8")
            if buf[-1] in ['\n', '\r']:
                break
            where = f.tell()
            byte = os.read(f.fileno(), 1)

        # print(buf, byte)

        if buf and buf[-1] in ['\n', '\r']:
            yield buf.strip()
            buf = ""
        else:
            f.seek(where)
            yield None




def main():
    print('starting in cwd = ', os.getcwd())
    print('data dir = ', PBAR_DATA_DIR)
    cmd = sys.argv[1:]
    cmd_string = ' '.join(cmd)
    print('cmd =', cmd_string)

    os.makedirs(PBAR_DATA_DIR, exist_ok=True)

    # Prepare by deleting the temp file if it exists
    try:
        os.remove(PBAR_TEMP_FILE)
    except OSError:
        pass

    model = Model()
    print("model =", model)
    model.train(cmd, read_program_logs())

    # Execute strace to get syscall duration
    strace_cmd = 'strace -fttt -e raw=all -o {PBAR_TEMP_FILE} {cmd}'.format(PBAR_TEMP_FILE=PBAR_TEMP_FILE, cmd=cmd_string)
    strace_proc = subprocess.Popen(strace_cmd, shell=True)
    print('strace cmd = ', strace_cmd)
    print('-' * 80)

    syscalls = []

    while not os.path.exists(PBAR_TEMP_FILE):
        time.sleep(1)

    last_report = None

    start_time = time.time_ns()
    report_logs = []
    with progressbar.ProgressBar(min_value=0, max_value=1, widgets=[progressbar.Percentage(), progressbar.Bar()]) as bar:
        with open(PBAR_TEMP_FILE, 'r') as strace_file:
            for line in line_generator(strace_file):
                strace_proc.poll()
                if strace_proc.returncode is not None and len(report_logs) > 0:
                    break

                if line:
                    call = SystemCall.parse(line.strip())
                    if call:
                        syscalls.append(call)

                if last_report is None or time.time_ns() - last_report > 0:
                    report_time = time.time_ns()
                    report_log = ProgramLog(cmd, syscalls)
                    report_logs.append(report_log)

                    last_report = report_time
                    completion_percentage = model.predict_completion(report_log)
                    bar.update(completion_percentage)

    write_program_log(ProgramLog(cmd, syscalls))

    # End by deleting the temp file
    os.remove(PBAR_TEMP_FILE)


if __name__ == '__main__':
    main()
