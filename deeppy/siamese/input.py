import cudarray as ca
from ..input import Input


class SiameseInput(Input):
    def __init__(self, x1, x2, batch_size=128):
        super(SiameseInput, self).__init__(x1, batch_size)
        if x1.shape[0] != x2.shape[0]:
            raise ValueError('shape mismatch between x1 and x2')
        self.x2 = x2

    def batches(self, phase=''):
        for batch_start, batch_stop in self._batch_slices():
            x1_batch = ca.array(self.x[batch_start:batch_stop])
            x2_batch = ca.array(self.x2[batch_start:batch_stop])
            yield x1_batch, x2_batch


class SupervisedSiameseInput(SiameseInput):
    def __init__(self, x1, x2, y, batch_size=128):
        super(SupervisedSiameseInput, self).__init__(x1, x2, batch_size)
        if x1.shape[0] != y.shape[0]:
            raise ValueError('shape mismatch between x and y')
        self.y = y

    def batches(self, phase='train'):
        if phase == 'train':
            for batch_start, batch_stop in self._batch_slices():
                x1_batch = ca.array(self.x[batch_start:batch_stop])
                x2_batch = ca.array(self.x2[batch_start:batch_stop])
                y_batch = ca.array(self.y[batch_start:batch_stop])
                yield x1_batch, x2_batch, y_batch
        elif phase == 'test':
            for x in super(SupervisedSiameseInput, self).batches():
                yield x

    @property
    def y_shape(self):
        return (self.batch_size,) + self.y.shape[1:]
