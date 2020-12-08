import numpy as np


class LearningSchedual(object):
    def __init__(self, optimizer, epochs, train_steps, lr):
        self.optimizer = optimizer
        self.train_steps = train_steps
        self.epochs = epochs  # !!!!!!!!!!!!!!!!!!!!!!!
        self.lr = lr

        self.warm_steps = 1
        self.all_steps_without_warm = epochs * train_steps - self.warm_steps
        self.optimizer.param_groups[0]['lr'] = self.lr[0] * 1 / self.warm_steps
        self.optimizer.param_groups[1]['lr'] = self.lr[1] * 1 / self.warm_steps

        print("init small lr:{} large lr:{}".format(self.optimizer.param_groups[0]['lr'],
                                                    self.optimizer.param_groups[1]['lr']))

    def update_lr(self, epoch, step):
        global_step = epoch * self.train_steps + step + 1
        global_step_without_warm_step = epoch * self.train_steps + step + 1 - self.warm_steps
        if global_step < self.warm_steps:
            self.optimizer.param_groups[0]['lr'] = self.lr[0] * global_step / self.warm_steps
            self.optimizer.param_groups[1]['lr'] = self.lr[1] * global_step / self.warm_steps
        elif global_step == self.warm_steps:
            self.optimizer.param_groups[0]['lr'] = self.lr[0]
            self.optimizer.param_groups[1]['lr'] = self.lr[1]
        elif step == 0:
            rate = (1 - global_step_without_warm_step / self.all_steps_without_warm)
            self.optimizer.param_groups[0]['lr'] = self.lr[0] * rate
            self.optimizer.param_groups[1]['lr'] = self.lr[1] * rate
        return self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[1]['lr']


class CosineAnnealingWithWarmRestart(object):
    def __init__(self, optimizer, epochs, train_steps, lr):
        self.optimizer = optimizer
        self.train_steps = train_steps
        self.epochs = epochs
        self.lr = lr

        self.warm_steps = 2400 # 2400
        self.all_steps_without_warm = epochs * train_steps - self.warm_steps
        self.optimizer.param_groups[0]['lr'] = self.lr[0] * 1 / self.warm_steps
        self.optimizer.param_groups[1]['lr'] = self.lr[1] * 1 / self.warm_steps

        print("init small lr:{} large lr:{}".format(self.optimizer.param_groups[0]['lr'],
                                                    self.optimizer.param_groups[1]['lr']))

    def update_lr(self, epoch, step):
        global_step = epoch * self.train_steps + step + 1
        global_step_without_warm_step = epoch * self.train_steps + step + 1 - self.warm_steps
        if global_step < self.warm_steps:
            self.optimizer.param_groups[0]['lr'] = self.lr[0] * global_step / self.warm_steps
            self.optimizer.param_groups[1]['lr'] = self.lr[1] * global_step / self.warm_steps
        elif global_step == self.warm_steps:
            self.optimizer.param_groups[0]['lr'] = self.lr[0]
            self.optimizer.param_groups[1]['lr'] = self.lr[1]
            print("small lr:{} large lr:{}".format(self.lr[0], self.lr[1]))
        elif global_step < self.train_steps and global_step_without_warm_step % 100 == 0:
            self.optimizer.param_groups[0]['lr'] = self.lr[0] * global_step_without_warm_step / (self.train_steps * self.epochs)
            self.optimizer.param_groups[1]['lr'] = self.lr[1] * global_step_without_warm_step / (self.train_steps * self.epochs)
        elif global_step % 10 == 0:
            cycle_step = self.train_steps * 4
            # lr_0 = (self.lr[0] - eta_min) * (1 - global_step / (self.train_steps * self.epochs)) + eta_min
            # lr_1 = (self.lr[1] - eta_min) * (1 - global_step / (self.train_steps * self.epochs)) + eta_min
            lr_0 = self.lr[0]
            lr_1 = self.lr[1]
            rate = np.cos(np.pi * global_step % cycle_step) + 1
            self.optimizer.param_groups[0]['lr'] = lr_0 * rate
            self.optimizer.param_groups[1]['lr'] = lr_1 * rate
        return self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[1]['lr']
