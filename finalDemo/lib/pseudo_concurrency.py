import time

class Timer:
    def set_waiting_time(self, seconds):
        self.waiting_time = seconds

    def set_executed_function(self, fun):
        self.fun = fun

    def __init__(self, fun, seconds, label=None, repetition_countdown=3):
        self.last_execution_time = 0
        self.fun = fun
        self.waiting_time = seconds
        self.label = label
        self.repetition_countdown = repetition_countdown
        self.initial_repetition_countdown = repetition_countdown

    def execute_if_ready(self, *args, **kvargs):
        current_time = time.time()
        ret = None
        if current_time - self.last_execution_time > self.waiting_time:
            try:
                if self.label:
                    print("Executing Timer '%s'..." % (self.label))
                ret = self.fun(*args, **kvargs)
            except Exception as e:
                print('Fehler:', e)
            self.last_execution_time = current_time
            return ret

    def execute_if_ready_repeatedly(self, *args, **kvargs):
        current_time = time.time()
        self.repetition_countdown -= 1
        ret = None
        if current_time - self.last_execution_time > self.waiting_time and \
            self.repetition_countdown == 0:
            try:
                if self.label:
                    print("Exectuing Timer '%s'..." % (self.label))
                self.repetition_countdown = self.initial_repetition_countdown
                ret = self.fun(*args, **kvargs)
            except Exception as e:
                print('Fehler:', e)
            self.last_execution_time = current_time
            return ret
