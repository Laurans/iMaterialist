import visdom
import time
import numpy as np

class Logger():
    def __init__(self, name):
        self.date_and_time = time.strftime('%d-%m-%Y--%H-%M-%S')

        visdom_opts = dict(env=name,server='http://localhost',port=8097)
        self.viz_ = visdom.Visdom(**visdom_opts)
        self.viz_dict_ = dict()


    def log_metrics(self, metric, tag, x, y):
        if metric not in self.viz_dict_.keys():
            self.viz_dict_[metric] = \
                self.viz_.line(Y=y, X=x,
                               opts={'legend': [tag],
                                    'title': metric,
                                    'xlabel': 'Epoch'})
        else:
            self.viz_.updateTrace(Y=y, X=x,
                                  name=tag,
                                  win=self.viz_dict_[metric],
                                  append=True)

class EarlyStopping():
    def __init__(self, min_delta, patience):
        self.min_delta = min_delta
        self.patience = patience
        self.previous_loss = np.Inf

    def on_train_begin(self):
        self.wait = 0
        self.stop = False


    def on_epoch_end(self, loss):
        if np.abs(self.previous_loss - loss) < self.min_delta:
            if self.wait >= self.patience:
                self.stop = True
                self.wait = 0
            else:
                self.wait +=1
        else:
            self.wait = 0

        self.previous_loss = loss

    def on_epoch_begin(self):
        return self.stop
