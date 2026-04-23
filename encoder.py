# encoder.py

class encoder:
    def __init__(self, bitstream):
        self.bitstream = bitstream
        self.low = 0x00000000
        self.high = 0xFFFFFFFF
        self.underflow_bits = 0
        self.HALF = 0x80000000
        self.FIRST_QTR = 0x40000000
        self.THIRD_QTR = 0xC0000000

    def encode(self, symbol, model):
        sym_low, sym_high = model.get_range(symbol)
        total = model.get_total()
        
        range_val = self.high - self.low + 1
        
        self.high = self.low + (range_val * sym_high) // total - 1
        self.low = self.low + (range_val * sym_low) // total

        while True:
            if self.high < self.HALF:
                self.bit_plus_follow(0)
            elif self.low >= self.HALF:
                self.bit_plus_follow(1)
                self.low -= self.HALF
                self.high -= self.HALF
            elif self.low >= self.FIRST_QTR and self.high < self.THIRD_QTR:
                self.underflow_bits += 1
                self.low -= self.FIRST_QTR
                self.high -= self.FIRST_QTR
            else:
                break

            self.low = (self.low << 1) & 0xFFFFFFFF
            self.high = ((self.high << 1) | 1) & 0xFFFFFFFF

    def bit_plus_follow(self, bit):
        self.bitstream.write_bit(bit)
        while self.underflow_bits > 0:
            self.bitstream.write_bit(bit ^ 1)
            self.underflow_bits -= 1

    def finish(self):
        self.underflow_bits += 1
        if self.low < self.FIRST_QTR:
            self.bit_plus_follow(0)
        else:
            self.bit_plus_follow(1)
        self.bitstream.flush()