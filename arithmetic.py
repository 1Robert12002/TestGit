# arithmetic.py
import argparse
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from bit_io import BitStream
from encoder import encoder
from decoder import decoder


class AdaptiveModel:
    def __init__(self, num_symbols, max_total=1 << 15):
        if num_symbols <= 0:
            raise ValueError("num_symbols must be positive")

        self.num_symbols = int(num_symbols)
        self.max_total = int(max_total)
        self.counts = [1] * self.num_symbols

    def get_total(self):
        return sum(self.counts)

    def get_range(self, symbol):
        symbol = int(symbol)
        if symbol < 0 or symbol >= self.num_symbols:
            raise ValueError(f"symbol out of range: {symbol}")

        low = 0
        for i in range(symbol):
            low += self.counts[i]
        high = low + self.counts[symbol]
        return low, high

    def update(self, symbol):
        symbol = int(symbol)
        if symbol < 0 or symbol >= self.num_symbols:
            raise ValueError(f"symbol out of range: {symbol}")

        self.counts[symbol] += 1

        if self.get_total() >= self.max_total:
            self.counts = [max(1, (count + 1) // 2) for count in self.counts]

class arithmetic:
    def __init__(self):
        self.NUM_SYMBOLS = 257  
        self.EOF_SYMBOL = 256   
        self.input_file = "test_input.txt"
        self.compressed_file = "compressed.bin"
        self.output_file = "test_output.txt"

    def setup_test_file(self, path=None):
        path = path or self.input_file
        with open(path, "w", encoding="utf-8") as f:
            f.write("Arithmetic coding relies on shifting and modeling! " * 10)
        print(f"Original file size: {os.path.getsize(path)} bytes")
        return path

    def run_encoder(self, input_file=None, compressed_file=None):
        input_file = input_file or self.input_file
        compressed_file = compressed_file or self.compressed_file

        with open(input_file, "rb") as f_in, open(compressed_file, "wb") as f_out:
            bstream_out = BitStream(f_out, 'w')
            enc = encoder(bstream_out)
            model = AdaptiveModel(self.NUM_SYMBOLS)
            
            while True:
                byte = f_in.read(1)
                if not byte:
                    break
                symbol = byte[0]
                enc.encode(symbol, model)
                model.update(symbol)
            
            enc.encode(self.EOF_SYMBOL, model)
            enc.finish()
            
        print(f"Compressed file size: {os.path.getsize(compressed_file)} bytes")
        return compressed_file

    def run_decoder(self, compressed_file=None, output_file=None):
        compressed_file = compressed_file or self.compressed_file
        output_file = output_file or self.output_file

        with open(compressed_file, "rb") as f_in, open(output_file, "wb") as f_out:
            bstream_in = BitStream(f_in, 'r')
            dec = decoder(bstream_in)
            model = AdaptiveModel(self.NUM_SYMBOLS) 
            
            while True:
                symbol = dec.decode_symbol(model)
                if symbol == self.EOF_SYMBOL:
                    break
                f_out.write(bytes([symbol]))
                model.update(symbol)
                
        print(f"Decompressed file size: {os.path.getsize(output_file)} bytes")
        return output_file

    def execute_all(self):
        """Runs the full pipeline."""
        self.setup_test_file()
        self.run_encoder()
        self.run_decoder()


class ArithmeticGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Arithmetic Coding Demo")
        self.app = arithmetic()

        self.input_path = tk.StringVar(value=self.app.input_file)
        self.compressed_path = tk.StringVar(value=self.app.compressed_file)
        self.output_path = tk.StringVar(value=self.app.output_file)
        self.status_var = tk.StringVar(value="Select a file or generate a test input.")

        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=12)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)

        ttk.Label(main, text="Input file:").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(main, textvariable=self.input_path, width=60).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Button(main, text="Browse", command=self._browse_input).grid(row=0, column=2, padx=(8, 0))

        ttk.Label(main, text="Compressed file:").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(main, textvariable=self.compressed_path, width=60).grid(row=1, column=1, sticky="ew", pady=4)
        ttk.Button(main, text="Browse", command=self._browse_compressed).grid(row=1, column=2, padx=(8, 0))

        ttk.Label(main, text="Output file:").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(main, textvariable=self.output_path, width=60).grid(row=2, column=1, sticky="ew", pady=4)
        ttk.Button(main, text="Browse", command=self._browse_output).grid(row=2, column=2, padx=(8, 0))

        button_row = ttk.Frame(main)
        button_row.grid(row=3, column=0, columnspan=3, sticky="w", pady=(10, 6))
        ttk.Button(button_row, text="Generate Test File", command=self._generate_test_file).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(button_row, text="Encode", command=self._encode).grid(row=0, column=1, padx=(0, 6))
        ttk.Button(button_row, text="Decode", command=self._decode).grid(row=0, column=2, padx=(0, 6))
        ttk.Button(button_row, text="Round Trip", command=self._round_trip).grid(row=0, column=3, padx=(0, 6))
        ttk.Button(button_row, text="Quit", command=self.root.destroy).grid(row=0, column=4)

        self.log = tk.Text(main, width=80, height=12, wrap="word")
        self.log.grid(row=4, column=0, columnspan=3, sticky="nsew", pady=(8, 6))
        main.rowconfigure(4, weight=1)

        ttk.Label(main, textvariable=self.status_var).grid(row=5, column=0, columnspan=3, sticky="w")

    def _browse_input(self):
        path = filedialog.askopenfilename(title="Select input file")
        if path:
            self.input_path.set(path)

    def _browse_compressed(self):
        path = filedialog.asksaveasfilename(title="Select compressed file", defaultextension=".bin")
        if path:
            self.compressed_path.set(path)

    def _browse_output(self):
        path = filedialog.asksaveasfilename(title="Select output file", defaultextension=".txt")
        if path:
            self.output_path.set(path)

    def _log(self, message):
        self.log.insert("end", message + "\n")
        self.log.see("end")

    def _set_paths(self):
        self.app.input_file = self.input_path.get().strip()
        self.app.compressed_file = self.compressed_path.get().strip()
        self.app.output_file = self.output_path.get().strip()

    def _generate_test_file(self):
        try:
            self._set_paths()
            path = self.app.setup_test_file(self.app.input_file)
            self._log(f"Generated test file: {path}")
            self.status_var.set(f"Test file created: {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Generate test file", str(exc))

    def _encode(self):
        try:
            self._set_paths()
            path = self.app.run_encoder(self.app.input_file, self.app.compressed_file)
            self._log(f"Encoded to: {path}")
            self.status_var.set(f"Compressed file written: {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Encode", str(exc))

    def _decode(self):
        try:
            self._set_paths()
            path = self.app.run_decoder(self.app.compressed_file, self.app.output_file)
            self._log(f"Decoded to: {path}")
            self.status_var.set(f"Output file written: {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Decode", str(exc))

    def _round_trip(self):
        try:
            self._set_paths()
            self.app.run_encoder(self.app.input_file, self.app.compressed_file)
            self.app.run_decoder(self.app.compressed_file, self.app.output_file)
            self._log("Round trip completed successfully.")
            self.status_var.set("Round trip completed successfully.")
        except Exception as exc:
            messagebox.showerror("Round trip", str(exc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arithmetic coding demo")
    parser.add_argument("--cli", action="store_true", help="Run the original command-line demo")
    args = parser.parse_args()

    if args.cli:
        arithmetic().execute_all()
    else:
        root = tk.Tk()
        ArithmeticGUI(root)
        root.mainloop()

