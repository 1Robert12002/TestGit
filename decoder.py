# decoder.py

class decoder:
    def __init__(self, bitstream):
        self.bitstream = bitstream
        self.low = 0x00000000
        self.high = 0xFFFFFFFF
        self.value = 0
        self.HALF = 0x80000000
        self.FIRST_QTR = 0x40000000
        self.THIRD_QTR = 0xC0000000

        for _ in range(32):
            self.value = (self.value << 1) | self._read_bit_or_zero()

    def _read_bit_or_zero(self):
        bit = self.bitstream.read_bit()
        return 0 if bit is None else int(bit)

    def decode_symbol(self, model):
        total = model.get_total()
        range_val = self.high - self.low + 1

        cum = ((self.value - self.low + 1) * total - 1) // range_val

        symbol = 0
        for i in range(model.num_symbols):
            _, sym_high = model.get_range(i)
            if sym_high > cum:
                symbol = i
                break

        sym_low, sym_high = model.get_range(symbol)
        self.high = self.low + (range_val * sym_high) // total - 1
        self.low = self.low + (range_val * sym_low) // total

        while True:
            if self.high < self.HALF:
                pass
            elif self.low >= self.HALF:
                self.low -= self.HALF
                self.high -= self.HALF
                self.value -= self.HALF
            elif self.low >= self.FIRST_QTR and self.high < self.THIRD_QTR:
                self.low -= self.FIRST_QTR
                self.high -= self.FIRST_QTR
                self.value -= self.FIRST_QTR
            else:
                break

            self.low = (self.low << 1) & 0xFFFFFFFF
            self.high = ((self.high << 1) | 1) & 0xFFFFFFFF
            self.value = ((self.value << 1) | self._read_bit_or_zero()) & 0xFFFFFFFF

        return symbol