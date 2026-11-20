class Scheduler:
    def __init__(
        self,
        learning_starts,
        train_freq,
        target_update_interval,
        log_interval = None
    ):
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.target_update_interval = target_update_interval
        self.log_interval = log_interval

        self.learning_started = False
        self.steps = 0
        self.grad_steps = 0

    def _passed_interval(self, cum_steps, diff_steps, interval):
        return cum_steps % interval + diff_steps >= interval

    def did_step(self, num_steps=1):
        self.steps += num_steps

        if not self.learning_started and self.steps >= self.learning_starts:
             self.learning_started = True

        if (
            self.learning_started and 
            self._passed_interval(self.steps, num_steps, self.train_freq)
        ):
            self._should_train = True
        
        if (
            self.log_interval is not None and
            self._passed_interval(self.steps, num_steps, self.log_interval)
        ):
            self._should_log = True 

    def did_grad_step(self, num_steps=1):
        self.grad_steps += num_steps

        if self._passed_interval(self.grad_steps, num_steps, self.target_update_interval):
            self._should_train_target = True


    def should_train(self):
        if self._should_train:
            self._should_train = False
            return True
        else:
            return False
    
    def should_log(self):
        if self._should_log:
            self._should_log = False
            return True
        else:
            return False
    
    def should_train_target(self):
        if self._should_train_target:
            self._should_train_target = False
            return True
        else:
            return False
        
    def reset(self):
        self.learning_started = False
        self._should_train = False
        self._should_log = False
        self._should_train_target = False
        self.steps = 0
        self.grad_steps = 0
