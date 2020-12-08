import time


class LogPrinter(object):
    def __init__(self, epochs, steps):
        self.epochs = epochs
        self.steps = steps
        self.begin_time = 0
        self.time_list_size = 250
        self.time_list = []

    @staticmethod
    def get_time_str(second):
        hour_unit = 60 * 60
        min_unit = 60
        h = int(second / hour_unit)
        second = second % hour_unit
        m = int(second / min_unit)
        s = second % min_unit

        time_string = ""
        if h > 0:
            time_string += "%d:%02d:%02d" % (h, m, s)
        elif m > 0:
            time_string += "%d:%02d" % (m, s)
        else:
            time_string += "%ds" % s
        return time_string

    def __call__(self, epoch, step, logs, new_line):
        epoch += 1
        step += 1
        log_str = ""
        log_keys = logs.keys()

        current_time = time.time()
        if step == 1:
            self.begin_time = current_time
            self.time_list = [current_time] * self.time_list_size
        self.time_list = self.time_list[1:]
        self.time_list.append(current_time)


        if step == 1:
            log_str += "\n\nEpoch: %4d/%-6d" % (epoch, self.epochs)
            log_str += '=' * (80 - len(log_str)) + '\n'

        if step == 1:
            _time = -1
        elif step == self.steps:
            # ALL
            _time = current_time - self.begin_time
        elif step < self.time_list_size:
            _time = (self.time_list[-1] - self.time_list[0]) * (self.steps - step) / (step - 1)
        else:
            # ETA
            _time = (self.time_list[-1] - self.time_list[0]) * (self.steps - step) / self.time_list_size

        log_str += '%d/%d' % (step, self.steps)
        log_str += ' - ETA: ' if step != self.steps else ' - ALL: '
        log_str += self.get_time_str(_time) if _time != -1 else "xx:xx:xx"

        # loss and metrics
        for key in log_keys:
            log_str += " - %s: " % key
            num = "{:.6f}".format(logs[key])
            log_str += num[:8]
        log_str = '\r' + log_str
        if new_line:
            log_str += '\n'

        print(log_str, end='')


class MovingData(object):
    def __init__(self, window):
        self.window = window
        self.data_dicts = {}
        self.moving_data = {}

    def __call__(self, globle_step, new_data):
        moving_index = int(globle_step % self.window)
        for key in new_data.keys():
            if key not in self.moving_data.keys():
                self.data_dicts[key] = [0] * self.window
                self.moving_data[key] = 0
            self.moving_data[key] += new_data[key] - self.data_dicts[key][moving_index]
            self.data_dicts[key][moving_index] = new_data[key]
        return self.moving_data

# #
# steps = 10
# epochs = 10
# log_printer = LogPrinter(epochs, steps, 1)
# log = {'loss': 9.1111, 'accuracy': 0.8766, 'val_loss': 2.222, 'val_accuracy': 0.1111}
# for j in range(epochs):
#     for i in range(steps):
#         log_printer(j, i, log)
#         time.sleep(1)
