import mindspore
from mindspore import ops
from mindspore.nn.learning_rate_schedule import LearningRateSchedule

class Noam(LearningRateSchedule):
    def __init__(self, start_lr, warmup_iter, end_iter):
        super().__init__()
        self.start_lr = start_lr
        self.warmup_iter = warmup_iter
        self.end_iter = end_iter

    def get_lr_warmup(self, num_iter):
        return self.start_lr / ops.sqrt(ops.scalar_to_tensor(self.warmup_iter)) * num_iter / self.warmup_iter
    
    def get_lr_decay(self, num_iter):
        return self.start_lr / ops.sqrt(num_iter)

    def construct(self, global_step):
        if global_step < self.warmup_iter:
            return self.get_lr_warmup(global_step)
        else:
            return self.get_lr_decay(global_step)
