from collections import defaultdict
import os
from math import nan
import numpy as np
from pathlib import Path
import pickle
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple
import uuid

PBAR_DATA_DIR = os.getenv("HOME") + '/.config/pbar'
PBAR_LOG_DIR = PBAR_DATA_DIR + '/logs'
PBAR_TEMP_FILE = '/tmp/pbar.out'


class SystemCall(object):
    def __init__(self, epoch_time: float, call_name: str, arguments: List[int], result: int):
        self.epoch_time = epoch_time
        self.call_name = call_name
        self.arguments = arguments
        self.result = result

    @staticmethod
    def parse(s: str):
        _pid, epoch_time, call = re.split('\\s\\s|\\s', s, maxsplit=2)

        if call.startswith('+') or call.startswith('-') or 'unfinished' in call or 'resumed' in call:
            return None

        epoch_time = float(epoch_time)
        name, rest = call.split('(', maxsplit=1)
        args_string, result_string = rest.split('=', maxsplit=1)

        args_string = args_string.strip().split(')')[0]
        result_string = result_string.split('(')[0]

        name = name.strip()
        if len(args_string) == 0:
            args = []
        else:
            args = [int(v, 0) for v in args_string.strip().split(',')]
        if result_string.strip() == '?':
            result_string = '0'
        result = int(result_string, 0)

        return SystemCall(epoch_time, name, args, result)


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
        feature_dict = defaultdict(int)
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
        self.model = KernelRidge(kernel='rbf')
        self.imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.trained = False

    @staticmethod
    def get_labeled_logs(dataset: List[ProgramLog]) -> List[Tuple[ProgramLog, float]]:
        items = []
        for log in dataset:
            total_time = log.duration()
            acc = []
            for syscall in log.calls:
                acc.append(syscall)
                new_log = ProgramLog(log.cmd, acc)
                items.append((new_log, (new_log.duration() / total_time) * 100))
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

        trimmed_dataset = Model.get_labeled_logs(trimmed_dataset)
        self.update_features([log for log, label in trimmed_dataset])

        if len(trimmed_dataset) > 0:
            x = []
            y = []
            for log, label in trimmed_dataset:
                x.append(self.extract_features(log))
                y.append(label)

            x = np.array(x)
            y = np.array(y)

            x = normalize(x)

            self.imp.fit(x, y)
            self.model.fit(x, y)
            self.trained = True

    def predict_completion(self, log: ProgramLog) -> float:
        if not self.trained:
            return 100.
        features = [self.extract_features(log)]
        features = self.imp.transform(features)

        return self.model.predict(normalize(features))


if __name__ == '__main__':
    print('pbar starting in', os.getcwd())
    cmd = sys.argv[1:]
    cmd_string = ' '.join(cmd)
    print('running:', cmd_string)

    # Prepare by deleting the temp file if it exists
    try:
        os.remove(PBAR_TEMP_FILE)
    except OSError:
        pass

    # Execute strace to get syscall duration
    strace_cmd = 'strace -fttt -e raw=all -o {PBAR_TEMP_FILE} {cmd}'.format(PBAR_TEMP_FILE=PBAR_TEMP_FILE, cmd=cmd_string)
    strace_proc = subprocess.Popen(strace_cmd, shell=True, stdout=subprocess.DEVNULL)
    print('strace:', strace_cmd)
    print('-' * 80)

    model = Model()
    model.train(cmd, read_program_logs())

    syscalls = []

    while not os.path.exists(PBAR_TEMP_FILE):
        time.sleep(1)

    with open(PBAR_TEMP_FILE, 'r') as strace_file:
        while True:
            where = strace_file.tell()
            line = strace_file.readline()
            if not line:
                strace_proc.poll()
                if strace_proc.returncode is not None:
                    break
                strace_file.seek(where)
            else:
                call = SystemCall.parse(line.strip())
                if call:
                    syscalls.append(call)
                completion_percentage = model.predict_completion(ProgramLog(cmd, syscalls))
                print(completion_percentage, '% done')

    write_program_log(ProgramLog(cmd, syscalls))

    # End by deleting the temp file
    os.remove(PBAR_TEMP_FILE)


