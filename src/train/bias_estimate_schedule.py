
class BiasEstimateScheduleBase(object):
    def get_value(self, epoch: int) -> float:
        raise NotImplementedError


class BiasEstimateLinearSchedule(BiasEstimateScheduleBase):
    def __init__(self, n_epochs: int):
        self._n_epochs = n_epochs
    
    def get_value(self, epoch: int) -> float:
        if self._n_epochs <= 1:
            return 1.0
        return epoch / (self._n_epochs - 1)


class BiasEstimateConstantSchedule(BiasEstimateScheduleBase):
    def __init__(self, value: float):
        self._value = value
    
    def get_value(self, epoch: int) -> float:
        return self._value