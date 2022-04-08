class Transducer:
    def __init__(self, **kwargs):
        self.tx_pos = kwargs.get('tx_pos', None)
        self.rx_pos = kwargs.get('rx_pos', None)
