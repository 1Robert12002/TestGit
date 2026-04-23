import os
import random

class BitStream:
    def __init__(self, file_handle, mode='r'):
        self.file = file_handle
        self.mode = mode
        self.buff = 0
        self.count = 0

    def write_bit(self, bit):
        # Rule: Store bit in Wbuff in position Wcnt
        if bit:
            self.buff |= (1 << (7 - (self.count % 8)))
        self.count += 1
        
        # Rule: if Wcnt multiple of 8, write to file
        if self.count % 8 == 0:
            self.file.write(bytes([self.buff]))
            self.buff = 0

    def read_bit(self):
        # Rule: if Rcnt multiple of 8, read from file
        if self.count % 8 == 0:
            byte_data = self.file.read(1)
            if not byte_data:
                return None
            self.buff = byte_data[0]
        
        # Rule: take bit from Rbuff from position Rcnt
        bit = (self.buff >> (7 - (self.count % 8))) & 1
        self.count += 1
        return bit

    def write_nbits(self, value, nr_bits):
        # Rule: for(NrBits-1 .... 0)
        for i in range(nr_bits - 1, -1, -1):
            bit = (value >> i) & 1
            self.write_bit(bit)

    def read_nbits(self, nr_bits):
        # Rule: for(NrBits-1 .... 0) add to val
        val = 0
        for i in range(nr_bits - 1, -1, -1):
            bit = self.read_bit()
            if bit is None: break
            if bit:
                val |= (1 << i)
        return val

    def flush(self):
        """Ensures the final partial byte is written at the end."""
        if 'w' in self.mode and self.count % 8 != 0:
            self.file.write(bytes([self.buff]))

# --- Implementation of the Whiteboard Test Logic (Right Column) ---

def copy_file_bit_by_bit(input_path, output_path):
    # 1. Open files in binary mode
    with open(input_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
        # 2. TNrB = 8 * size of input file
        file_size = os.path.getsize(input_path)
        tnrb = 8 * file_size
        
        reader = BitStream(f_in, 'r')
        writer = BitStream(f_out, 'w')

        # 3. do { ... } while(TNrB > 0)
        while tnrb > 0:
            # nrB = random [1...32]
            nrb = random.randint(1, 32)
            
            # if(nrB > TNrB) nrB = TNrB
            if nrb > tnrb:
                nrb = tnrb
            
            # b = ReadNBITS(nrB)
            # WriteNBITS(b, nrB)
            val = reader.read_nbits(nrb)
            writer.write_nbits(val, nrb)
            
            # TNrB -= nrB
            tnrb -= nrb
        
        writer.flush()
    print(f"Transfer complete. {file_size} bytes processed.")

# Example usage:
# copy_file_bit_by_bit('source.txt', 'destination.txt')