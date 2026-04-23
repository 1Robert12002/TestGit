import numpy as np
import io
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from bit_io import BitStream


def _clip_uint8(value):
    return max(0, min(255, int(value)))


def get_prediction(matrix, row, col, predictor_type):
    """
    Compute prediction for `matrix[row, col]` using predictor mode 0..8.

    Neighbor definitions:
    - A = (row, col-1)
    - B = (row-1, col)
    - C = (row-1, col-1)

    Boundary rules:
    - first pixel (0, 0): 128
    - first line (row == 0): A
    - first column (col == 0): B
    """
    if row == 0 and col == 0:
        return 128

    if row == 0:
        return _clip_uint8(matrix[row, col - 1])

    if col == 0:
        return _clip_uint8(matrix[row - 1, col])

    a = int(matrix[row, col - 1])
    b = int(matrix[row - 1, col])
    c = int(matrix[row - 1, col - 1])

    if predictor_type == 0:
        pred = 128
    elif predictor_type == 1:
        pred = a
    elif predictor_type == 2:
        pred = b
    elif predictor_type == 3:
        pred = c
    elif predictor_type == 4:
        pred = a + b - c
    elif predictor_type == 5:
        pred = (a + b - c) // 2
    elif predictor_type == 6:
        pred = b + ((a - c) // 2)
    elif predictor_type == 7:
        pred = (a + b) // 2
    elif predictor_type == 8:
        # JPEG-LS predictor
        if c >= max(a, b):
            pred = min(a, b)
        elif c <= min(a, b):
            pred = max(a, b)
        else:
            pred = a + b - c
    else:
        raise ValueError("predictor_type must be in [0..8]")

    return _clip_uint8(pred)


def coder_loop(matrix, predictor_type, k):
    """
    Predictive coder with feedback for a 256x256 grayscale image.

    For each pixel:
      ep  = original - prediction
      epq = floor((ep + k) / (2k + 1))
      epd = epq * (2k + 1)
      r   = prediction + epd

    Important:
      Predictions are computed from reconstructed samples (feedback),
      not from original samples.

    Returns:
      (quantized_error, reconstructed)
    """
    if not (0 <= int(k) <= 10):
        raise ValueError("k (accepted error) must be in [0..10]")

    src = np.asarray(matrix, dtype=np.uint8)
    if src.shape != (256, 256):
        raise ValueError(f"matrix must have shape (256, 256), got {src.shape}")

    q_err = np.zeros((256, 256), dtype=np.int16)
    recon = np.zeros((256, 256), dtype=np.uint8)

    step = 2 * int(k) + 1

    for row in range(256):
        for col in range(256):
            pred = get_prediction(recon, row, col, predictor_type)

            ep = int(src[row, col]) - int(pred)
            epq = (ep + int(k)) // step
            epd = epq * step

            r = _clip_uint8(int(pred) + epd)

            q_err[row, col] = epq
            recon[row, col] = r

    return q_err, recon


def coder_loop_with_prediction_error(matrix, predictor_type, k):
    """Return prediction error, quantized prediction error, and reconstructed image."""
    if not (0 <= int(k) <= 10):
        raise ValueError("k (accepted error) must be in [0..10]")

    src = np.asarray(matrix, dtype=np.uint8)
    if src.shape != (256, 256):
        raise ValueError(f"matrix must have shape (256, 256), got {src.shape}")

    pred_err = np.zeros((256, 256), dtype=np.int16)
    q_err = np.zeros((256, 256), dtype=np.int16)
    recon = np.zeros((256, 256), dtype=np.uint8)

    step = 2 * int(k) + 1

    for row in range(256):
        for col in range(256):
            pred = get_prediction(recon, row, col, predictor_type)
            ep = int(src[row, col]) - int(pred)
            epq = (ep + int(k)) // step
            epd = epq * step

            pred_err[row, col] = ep
            q_err[row, col] = epq
            recon[row, col] = _clip_uint8(int(pred) + epd)

    return pred_err, q_err, recon


def epq_to_category_index(epq):
    """
    Map quantized error `e_pq` to (category, index) for JPEG Table coding.

    Categories are 0..8:
      cat=0: {0}
      cat>=1: values with magnitude in [2^(cat-1), 2^cat - 1]

    Index follows JPEG signed-amplitude convention on each line.
    """
    v = int(epq)

    if v == 0:
        return 0, 0

    mag = abs(v)
    category = mag.bit_length()
    if category > 8:
        raise ValueError(f"e_pq={v} out of supported Mode T range for categories 0..8")

    if v > 0:
        index = v
    else:
        index = ((1 << category) - 1) + v

    return category, index


def category_index_to_epq(category, index):
    """Reverse mapping from (category, index) back to quantized error `e_pq`."""
    c = int(category)
    i = int(index)

    if c < 0 or c > 8:
        raise ValueError("category must be in [0..8]")

    if c == 0:
        if i != 0:
            raise ValueError("index must be 0 when category is 0")
        return 0

    max_index = (1 << c) - 1
    if i < 0 or i > max_index:
        raise ValueError(f"index out of range for category {c}: {i}")

    threshold = 1 << (c - 1)
    if i >= threshold:
        return i
    return i - max_index


def write_mode_t_value(writer, epq):
    """
    Write one quantized error using Mode T:
      1) unary(category): category ones followed by a zero
      2) `category` bits for index (no bits for category 0)
    """
    category, index = epq_to_category_index(epq)

    for _ in range(category):
        writer.write_bit(1)
    writer.write_bit(0)

    if category > 0:
        writer.write_nbits(index, category)


def read_mode_t_value(reader):
    """
    Read one quantized error encoded with Mode T.

    Reverse of `write_mode_t_value`:
      - read unary until first 0 -> category
      - read `category` bits -> index
      - reconstruct e_pq
    """
    category = 0
    while True:
        bit = reader.read_bit()
        if bit is None:
            raise EOFError("Unexpected EOF while reading unary category")
        if bit == 0:
            break
        category += 1
        if category > 8:
            raise ValueError("Invalid category in stream (expected 0..8)")

    index = 0
    if category > 0:
        index = reader.read_nbits(category)

    return category_index_to_epq(category, index)


def write_mode_t_matrix(bit_writer, q_err_matrix):
    """Write an entire 256x256 quantized-error matrix using Mode T coding."""
    q = np.asarray(q_err_matrix)
    if q.shape != (256, 256):
        raise ValueError(f"q_err_matrix must have shape (256, 256), got {q.shape}")

    for row in range(256):
        for col in range(256):
            write_mode_t_value(bit_writer, int(q[row, col]))


def read_mode_t_matrix(bit_reader, rows=256, cols=256):
    """Read a quantized-error matrix encoded with Mode T coding."""
    out = np.zeros((rows, cols), dtype=np.int16)
    for row in range(rows):
        for col in range(cols):
            out[row, col] = read_mode_t_value(bit_reader)
    return out


def _arithmetic_encode_uniform(symbols, alphabet_size):
    """Basic arithmetic encoder for symbols in [0, alphabet_size-1]."""
    full = 1 << 32
    half = full >> 1
    quarter = half >> 1
    three_quarter = quarter * 3

    low = 0
    high = full - 1
    pending = 0

    out_buff = io.BytesIO()
    writer = BitStream(out_buff, mode='w')

    def output_bit_plus_pending(bit):
        nonlocal pending
        writer.write_bit(bit)
        for _ in range(pending):
            writer.write_bit(1 - bit)
        pending = 0

    total = int(alphabet_size)
    for s in symbols:
        sym = int(s)
        if sym < 0 or sym >= total:
            raise ValueError(f"Symbol out of range: {sym}")

        rng = high - low + 1
        high = low + (rng * (sym + 1)) // total - 1
        low = low + (rng * sym) // total

        while True:
            if high < half:
                output_bit_plus_pending(0)
            elif low >= half:
                output_bit_plus_pending(1)
                low -= half
                high -= half
            elif low >= quarter and high < three_quarter:
                pending += 1
                low -= quarter
                high -= quarter
            else:
                break

            low = (low << 1) & (full - 1)
            high = ((high << 1) & (full - 1)) | 1

    pending += 1
    if low < quarter:
        output_bit_plus_pending(0)
    else:
        output_bit_plus_pending(1)

    writer.flush()
    return out_buff.getvalue()


def _arithmetic_decode_uniform(data_bytes, n_symbols, alphabet_size):
    """Basic arithmetic decoder for symbols in [0, alphabet_size-1]."""
    full = 1 << 32
    half = full >> 1
    quarter = half >> 1
    three_quarter = quarter * 3

    low = 0
    high = full - 1

    in_buff = io.BytesIO(data_bytes)
    reader = BitStream(in_buff, mode='r')

    def read_bit_or_zero():
        bit = reader.read_bit()
        return 0 if bit is None else int(bit)

    code = 0
    for _ in range(32):
        code = (code << 1) | read_bit_or_zero()

    total = int(alphabet_size)
    decoded = np.zeros(n_symbols, dtype=np.int16)

    for i in range(n_symbols):
        rng = high - low + 1
        value = ((code - low + 1) * total - 1) // rng
        if value < 0:
            value = 0
        elif value >= total:
            value = total - 1

        sym = int(value)
        decoded[i] = sym

        high = low + (rng * (sym + 1)) // total - 1
        low = low + (rng * sym) // total

        while True:
            if high < half:
                pass
            elif low >= half:
                low -= half
                high -= half
                code -= half
            elif low >= quarter and high < three_quarter:
                low -= quarter
                high -= quarter
                code -= quarter
            else:
                break

            low = (low << 1) & (full - 1)
            high = ((high << 1) & (full - 1)) | 1
            code = ((code << 1) & (full - 1)) | read_bit_or_zero()

    return decoded


def encode_mode_a_matrix(q_err_matrix):
    """
    Mode A arithmetic coding.
    Lab-required mapping: shift quantized errors by +255.
    """
    q = np.asarray(q_err_matrix, dtype=np.int16)
    if q.shape != (256, 256):
        raise ValueError(f"q_err_matrix must have shape (256, 256), got {q.shape}")

    shifted = q.astype(np.int32) + 255
    if shifted.min() < 0 or shifted.max() > 510:
        raise ValueError("Mode A supports quantized errors only in [-255, 255]")

    return _arithmetic_encode_uniform(shifted.ravel(), alphabet_size=511)


def decode_mode_a_matrix(payload_bytes, rows=256, cols=256):
    """Decode Mode A arithmetic-coded payload into quantized errors."""
    n = int(rows) * int(cols)
    shifted = _arithmetic_decode_uniform(payload_bytes, n_symbols=n, alphabet_size=511)
    q = shifted.astype(np.int16) - 255
    return q.reshape((rows, cols))


def encode_error_payload(q_err_matrix, mode):
    """Encode quantized error matrix payload for mode F/T/A."""
    q = np.asarray(q_err_matrix, dtype=np.int16)
    mode = str(mode).upper()

    if mode == "F":
        return q.tobytes()

    if mode == "T":
        buff = io.BytesIO()
        writer = BitStream(buff, mode='w')
        write_mode_t_matrix(writer, q)
        writer.flush()
        return buff.getvalue()

    if mode == "A":
        return encode_mode_a_matrix(q)

    raise ValueError("mode must be one of: F, T, A")


def decode_error_payload(payload_bytes, mode, rows=256, cols=256):
    """Decode quantized error payload for mode F/T/A."""
    mode = str(mode).upper()
    n = int(rows) * int(cols)

    if mode == "F":
        expected = n * 2
        if len(payload_bytes) < expected:
            raise ValueError(f"Invalid F payload size: expected at least {expected} bytes")
        return np.frombuffer(payload_bytes[:expected], dtype=np.int16).reshape((rows, cols)).copy()

    if mode == "T":
        buff = io.BytesIO(payload_bytes)
        reader = BitStream(buff, mode='r')
        return read_mode_t_matrix(reader, rows, cols)

    if mode == "A":
        return decode_mode_a_matrix(payload_bytes, rows, cols)

    raise ValueError("mode must be one of: F, T, A")


def decode_from_quantized_error(q_err_matrix, predictor_type, k):
    """Decode (reconstruct) image from quantized errors using feedback prediction."""
    q = np.asarray(q_err_matrix, dtype=np.int16)
    if q.shape != (256, 256):
        raise ValueError(f"q_err_matrix must have shape (256, 256), got {q.shape}")

    recon = np.zeros((256, 256), dtype=np.uint8)
    step = 2 * int(k) + 1

    for row in range(256):
        for col in range(256):
            pred = get_prediction(recon, row, col, predictor_type)
            epd = int(q[row, col]) * step
            recon[row, col] = _clip_uint8(int(pred) + epd)

    return recon


def mode_roundtrip(q_err_matrix, mode):
    """
    Simulate save/load roundtrip for quantized errors according to mode:
      F: fixed 16-bit signed values
      T: JPEG table coding with BitStream
      A: ASCII text values
    """
    payload = encode_error_payload(q_err_matrix, mode)
    return decode_error_payload(payload, mode, 256, 256)


class CompressionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Predictive Image Codec Lab")

        self.bmp = None
        self.source_bmp_path = None
        self.source_prd_path = None
        self.original = None
        self.pred_error = None
        self.q_err = None
        self.decoded = None
        self.error_display = None

        self.photo_original = None
        self.photo_error = None
        self.photo_decoded = None

        self._build_ui()

    def _build_ui(self):
        controls = ttk.Frame(self.root, padding=8)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(20, weight=1)

        ttk.Button(controls, text="Load BMP", command=self.load_bmp).grid(row=0, column=0, padx=4)
        ttk.Button(controls, text="Load Encoded", command=self.load_encoded).grid(row=0, column=1, padx=4)

        ttk.Label(controls, text="Predictor:").grid(row=0, column=2, padx=(12, 4))
        self.predictor_var = tk.StringVar(value="8: JpegLS")
        self.predictor_combo = ttk.Combobox(
            controls,
            width=16,
            state="readonly",
            textvariable=self.predictor_var,
            values=[
                "0: 128",
                "1: A",
                "2: B",
                "3: C",
                "4: A+B-C",
                "5: (A+B-C)/2",
                "6: B+(A-C)/2",
                "7: (A+B)/2",
                "8: JpegLS",
            ],
        )
        self.predictor_combo.grid(row=0, column=3, padx=4)

        ttk.Label(controls, text="k:").grid(row=0, column=4, padx=(12, 4))
        self.k_var = tk.IntVar(value=2)
        ttk.Spinbox(controls, from_=0, to=10, textvariable=self.k_var, width=5).grid(row=0, column=5, padx=4)

        ttk.Label(controls, text="Save Mode:").grid(row=0, column=6, padx=(12, 4))
        self.mode_var = tk.StringVar(value="T")
        ttk.Combobox(
            controls,
            width=4,
            state="readonly",
            textvariable=self.mode_var,
            values=["F", "T", "A"],
        ).grid(row=0, column=7, padx=4)

        ttk.Label(controls, text="Contrast:").grid(row=0, column=8, padx=(12, 4))
        self.scale_var = tk.DoubleVar(value=4.0)
        ttk.Spinbox(controls, from_=0.1, to=64.0, increment=0.1, textvariable=self.scale_var, width=6).grid(row=0, column=9, padx=4)

        ttk.Button(controls, text="Process", command=self.process).grid(row=0, column=10, padx=(12, 4))
        ttk.Button(controls, text="Compute Error", command=self.compute_error).grid(row=0, column=11, padx=4)
        ttk.Button(controls, text="Save Encoded", command=self.save_encoded).grid(row=0, column=12, padx=4)
        ttk.Button(controls, text="Save Decoded BMP", command=self.save_decoded_bmp).grid(row=0, column=13, padx=4)

        error_frame = ttk.LabelFrame(controls, text="Error Inspection", padding=6)
        error_frame.grid(row=1, column=0, columnspan=14, sticky="ew", pady=(8, 0))

        ttk.Label(error_frame, text="Error Image Source:").grid(row=0, column=0, padx=(0, 4))
        self.error_source_var = tk.StringVar(value="Quantized Prediction Error")
        ttk.Combobox(
            error_frame,
            width=28,
            state="readonly",
            textvariable=self.error_source_var,
            values=["Prediction Error", "Quantized Prediction Error"],
        ).grid(row=0, column=1, padx=4)

        ttk.Label(error_frame, text="Contrast:").grid(row=0, column=2, padx=(12, 4))
        ttk.Spinbox(error_frame, from_=0.1, to=64.0, increment=0.1, textvariable=self.scale_var, width=6).grid(row=0, column=3, padx=4)

        ttk.Button(error_frame, text="Refresh", command=self.refresh_error_image).grid(row=0, column=4, padx=(12, 4))

        self.status_var = tk.StringVar(value="Load a 256x256 grayscale BMP to begin.")
        ttk.Label(controls, textvariable=self.status_var).grid(row=2, column=0, columnspan=21, sticky="w", pady=(8, 0))

        images_frame = ttk.Frame(self.root, padding=8)
        images_frame.grid(row=1, column=0, sticky="nsew")
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

        for i in range(3):
            images_frame.columnconfigure(i, weight=1)

        self._build_image_panel(images_frame, 0, "Original", "original")
        self._build_image_panel(images_frame, 1, "Error", "error")
        self._build_image_panel(images_frame, 2, "Decoded", "decoded")

        hist_frame = ttk.LabelFrame(self.root, text="histrograma", padding=8)
        hist_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        hist_frame.columnconfigure(1, weight=1)

        ttk.Label(hist_frame, text="Histogram Source:").grid(row=0, column=0, padx=(0, 4), sticky="w")
        self.hist_source_var = tk.StringVar(value="Coder - Q Prediction Error Image")
        ttk.Combobox(
            hist_frame,
            width=34,
            state="readonly",
            textvariable=self.hist_source_var,
            values=[
                "Coder - Orig Image",
                "Coder - Prediction Error Image",
                "Coder - Q Prediction Error Image",
                "Coder - Decoded Image",
                "Decoder - Q Prediction Error Image",
                "Decoder - DQ Prediction Error Image",
                "Decoder - Decoded Image",
            ],
        ).grid(row=0, column=1, padx=4, sticky="w")

        ttk.Label(hist_frame, text="scale:").grid(row=0, column=2, padx=(12, 4), sticky="w")
        self.hist_scale_var = tk.DoubleVar(value=0.01)
        ttk.Spinbox(
            hist_frame,
            from_=0.0,
            to=100.0,
            increment=0.01,
            textvariable=self.hist_scale_var,
            width=8,
        ).grid(row=0, column=3, padx=4, sticky="w")

        ttk.Button(hist_frame, text="Histogram Refresh", command=self.refresh_histogram).grid(row=0, column=4, padx=(12, 4))

        tk.Label(hist_frame, text="NO Auto-scaling", fg="red").grid(row=0, column=5, padx=(8, 0), sticky="w")

        self.hist_canvas = tk.Canvas(hist_frame, width=511, height=180, bg="white", highlightthickness=1)
        self.hist_canvas.grid(row=1, column=0, columnspan=6, sticky="ew", pady=(8, 0))

    def _build_image_panel(self, parent, column, title, key):
        frame = ttk.LabelFrame(parent, text=title, padding=6)
        frame.grid(row=0, column=column, padx=6, sticky="n")
        label = ttk.Label(frame)
        label.grid(row=0, column=0)
        setattr(self, f"{key}_label", label)

    def _get_selected_predictor(self):
        value = str(self.predictor_var.get()).strip()
        if ":" in value:
            value = value.split(":", 1)[0].strip()
        p = int(value)
        if p < 0 or p > 8:
            raise ValueError("Predictor must be in [0..8]")
        return p

    def _get_dq_error_matrix(self):
        if self.q_err is None:
            return None
        step = 2 * int(self.k_var.get()) + 1
        return self.q_err.astype(np.int16) * step

    def _get_error_source_matrix(self, source_name):
        source_name = str(source_name)
        if source_name == "Prediction Error":
            return self.pred_error
        if source_name == "Quantized Prediction Error":
            return self.q_err
        if source_name in ("Original Image", "Coder - Orig Image"):
            return self.original
        if source_name in ("Decoded Image", "Coder - Decoded Image", "Decoder - Decoded Image"):
            return self.decoded
        if source_name == "Coder - Prediction Error Image":
            return self.pred_error
        if source_name in ("Coder - Q Prediction Error Image", "Decoder - Q Prediction Error Image"):
            return self.q_err
        if source_name == "Decoder - DQ Prediction Error Image":
            return self._get_dq_error_matrix()
        raise ValueError(f"Unknown source: {source_name}")

    def _matrix_to_histogram_values(self, matrix, source_name):
        arr = np.asarray(matrix)
        if source_name in (
            "Original Image",
            "Decoded Image",
            "Coder - Orig Image",
            "Coder - Decoded Image",
            "Decoder - Decoded Image",
        ):
            return arr.astype(np.int16).ravel()
        return np.clip(arr.astype(np.int16), -255, 255).ravel()

    def _get_error_image_matrix(self):
        source_name = self.error_source_var.get()
        matrix = self._get_error_source_matrix(source_name)
        if matrix is None:
            raise ValueError(f"{source_name} is not available yet.")
        scale = float(self.scale_var.get())
        error_values = np.asarray(matrix, dtype=np.int16)
        return np.clip(error_values * scale + 128, 0, 255).astype(np.uint8)

    def refresh_error_image(self):
        try:
            if self.error_source_var.get() == "Prediction Error" and self.pred_error is None:
                raise ValueError("Prediction Error is not available. Load a BMP and click Process first.")
            if self.error_source_var.get() == "Quantized Prediction Error" and self.q_err is None:
                raise ValueError("Quantized Prediction Error is not available. Load a BMP and click Process first.")
            self.error_display = self._get_error_image_matrix()
            self._update_images()
            self.status_var.set(
                f"Error image refreshed from {self.error_source_var.get()} with scale={self.scale_var.get()}"
            )
        except Exception as exc:
            messagebox.showerror("Error refresh", str(exc))

    def refresh_histogram(self):
        try:
            self._draw_histogram()
            self.status_var.set(
                f"Histogram refreshed from {self.hist_source_var.get()} with scale={self.hist_scale_var.get()}"
            )
        except Exception as exc:
            messagebox.showerror("Histogram refresh", str(exc))

    def _matrix_to_photo(self, matrix):
        arr = np.asarray(matrix, dtype=np.uint8)
        if arr.shape != (256, 256):
            raise ValueError(f"Expected image shape (256,256), got {arr.shape}")

        photo = tk.PhotoImage(width=256, height=256)
        for y in range(256):
            row = arr[y]
            row_colors = " ".join(f"#{v:02x}{v:02x}{v:02x}" for v in row)
            photo.put("{" + row_colors + "}", to=(0, y))
        return photo

    def _update_images(self):
        if self.original is not None:
            self.photo_original = self._matrix_to_photo(self.original)
            self.original_label.configure(image=self.photo_original)

        if self.error_display is not None:
            self.photo_error = self._matrix_to_photo(self.error_display)
            self.error_label.configure(image=self.photo_error)

        if self.decoded is not None:
            self.photo_decoded = self._matrix_to_photo(self.decoded)
            self.decoded_label.configure(image=self.photo_decoded)

    def _draw_histogram(self):
        self.hist_canvas.delete("all")

        source_name = self.hist_source_var.get()
        matrix = self._get_error_source_matrix(source_name)
        if matrix is None:
            raise ValueError(f"{source_name} is not available yet.")

        vals = self._matrix_to_histogram_values(matrix, source_name)
        vals = vals[(vals >= -255) & (vals <= 255)]
        counts = np.bincount(vals + 255, minlength=511)

        w = 511
        h = 180
        self.hist_canvas.config(width=w, height=h)

        scale = float(self.hist_scale_var.get())
        if scale < 0:
            raise ValueError("Histogram Scale must be non-negative")

        if counts.size == 0 or int(counts.max()) == 0:
            return

        for x in range(511):
            bar_h = min(h, int(counts[x] * scale))
            self.hist_canvas.create_line(x, h, x, h - bar_h, fill="#204a87")

        self.hist_canvas.create_line(255, 0, 255, h, fill="#cc0000")

    def load_bmp(self):
        path = filedialog.askopenfilename(
            title="Select 256x256 grayscale BMP",
            filetypes=[("BMP files", "*.bmp"), ("All files", "*.*")],
        )
        if not path:
            return
        

        try:
            bmp = GrayscaleBMP256()
            bmp.read(path)
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            return

        self.bmp = bmp
        self.source_bmp_path = path
        self.source_prd_path = None
        self.original = bmp.pixels.copy()
        self.pred_error = None
        self.q_err = None
        self.decoded = None
        self.error_display = np.full((256, 256), 128, dtype=np.uint8)
        self._update_images()
        self.hist_canvas.delete("all")
        self.status_var.set(f"Loaded: {Path(path).name}")

    def load_encoded(self):
        path = filedialog.askopenfilename(
            title="Load encoded NL",
            filetypes=[("NL files", "*.nl"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            with open(path, "rb") as f:
                header = f.read(1078)
                if len(header) != 1078:
                    raise ValueError("Invalid .nl file: missing 1078-byte BMP header")

                meta = f.read(3)
                if len(meta) != 3:
                    raise ValueError("Invalid .nl file: missing predictor/k/mode bytes")

                predictor = int(meta[0])
                k = int(meta[1])
                mode = chr(meta[2]).upper()
                payload = f.read()

            if predictor < 0 or predictor > 8:
                raise ValueError(f"Invalid predictor in .prd: {predictor}")
            if k < 0 or k > 10:
                raise ValueError(f"Invalid k in .prd: {k}")
            if mode not in ("F", "T", "A"):
                raise ValueError(f"Invalid mode in .prd: {mode}")

            q_err = decode_error_payload(payload, mode, 256, 256)
            decoded = decode_from_quantized_error(q_err, predictor, k)

            bmp = GrayscaleBMP256()
            bmp.header = header
            bmp._bottom_up = bmp._read_bmp_height() > 0
            bmp.pixels = decoded.copy()

            scale = int(self.scale_var.get())
            err_img = np.clip(q_err.astype(np.int32) * scale + 128, 0, 255).astype(np.uint8)

            self.bmp = bmp
            self.source_prd_path = path
            self.source_bmp_path = None
            self.pred_error = None
            self.q_err = q_err
            self.decoded = decoded
            self.error_display = err_img
            self.original = decoded.copy()

            self.predictor_var.set(f"{predictor}: {self.predictor_combo['values'][predictor].split(':', 1)[1].strip()}")
            self.k_var.set(k)
            self.mode_var.set(mode)

            self._update_images()
            self._draw_histogram()
            self.status_var.set(
                f"Loaded NL: {Path(path).name} (predictor={predictor}, k={k}, mode={mode})"
            )
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))

    def process(self):
        if self.original is None:
            messagebox.showwarning("Missing image", "Please load a BMP image first.")
            return

        try:
            predictor = self._get_selected_predictor()
            k = int(self.k_var.get())
            mode = self.mode_var.get().upper()
            scale = float(self.scale_var.get())

            pred_err, q_err, _ = coder_loop_with_prediction_error(self.original, predictor, k)
            q_from_mode = mode_roundtrip(q_err, mode)
            decoded = decode_from_quantized_error(q_from_mode, predictor, k)

            err_img = np.clip(q_from_mode.astype(np.int32) * scale + 128, 0, 255).astype(np.uint8)

            self.pred_error = pred_err
            self.q_err = q_from_mode
            self.decoded = decoded
            self.error_display = err_img

            self.refresh_error_image()
            self._update_images()
            self._draw_histogram()
            self.status_var.set(
                f"Processed with predictor={predictor}, k={k}, mode={mode}, scale={scale}"
            )
        except Exception as exc:
            messagebox.showerror("Processing error", str(exc))

    def compute_error(self):
        if self.original is None or self.decoded is None:
            messagebox.showwarning("Missing data", "Load and process an image first.")
            return

        diff = self.original.astype(np.int16) - self.decoded.astype(np.int16)
        dmin = int(diff.min())
        dmax = int(diff.max())
        messagebox.showinfo("Compute Error (Orig - Decoded)", f"Orig - Decoded\nMin: {dmin}\nMax: {dmax}")

    def save_encoded(self):
        if self.q_err is None or self.bmp is None or self.bmp.header is None:
            messagebox.showwarning("Missing data", "Load and process an image first.")
            return

        predictor = self._get_selected_predictor()
        k = int(self.k_var.get())
        mode = self.mode_var.get().upper()

        base_name = "image.bmp"
        if self.source_bmp_path:
            base_name = Path(self.source_bmp_path).name
        elif self.source_prd_path:
            nl_name = Path(self.source_prd_path).name
            marker = ".bmp.k"
            pos = nl_name.find(marker)
            base_name = nl_name[: pos + 4] if pos != -1 else "image.bmp"

        suggested = f"{base_name}.k{k}p{predictor}{mode}.nl"
        path = filedialog.asksaveasfilename(
            title=f"Save encoded NL (mode {mode})",
            initialfile=suggested,
            defaultextension=".nl",
            filetypes=[("NL files", "*.nl"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            payload = encode_error_payload(self.q_err, mode)

            with open(path, "wb") as f:
                f.write(self.bmp.header)
                f.write(bytes([predictor & 0xFF]))
                f.write(bytes([k & 0xFF]))
                f.write(mode.encode("ascii"))
                f.write(payload)

            self.status_var.set(f"Encoded NL saved ({mode}) -> {Path(path).name}")
            self.source_prd_path = path
        except Exception as exc:
            messagebox.showerror("Save error", str(exc))

    def save_decoded_bmp(self):
        if self.bmp is None or self.decoded is None:
            messagebox.showwarning("Missing data", "Load and process an image first.")
            return

        path = filedialog.asksaveasfilename(
            title="Save decoded BMP",
            defaultextension=".bmp",
            filetypes=[("BMP files", "*.bmp"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            self.bmp.write(path, self.decoded)
            self.status_var.set(f"Decoded BMP saved -> {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Save error", str(exc))


class GrayscaleBMP256:
    """
    Handler for 256x256 8-bit grayscale BMP images.

    Format assumptions for this lab:
    - First 1078 bytes are BMP + DIB + 256-color palette header
    - Pixel payload is exactly 256 * 256 bytes
    """

    HEADER_SIZE = 1078
    WIDTH = 256
    HEIGHT = 256
    PIXEL_COUNT = WIDTH * HEIGHT

    def __init__(self):
        self.header = None
        self.pixels = None
        self._bottom_up = True

    def _read_bmp_height(self):
        """Read signed BMP height from DIB header (offset 22, 4 bytes LE)."""
        if self.header is None or len(self.header) < 26:
            return self.HEIGHT
        return int.from_bytes(self.header[22:26], byteorder="little", signed=True)

    def read(self, path):
        """Read BMP header and pixels into memory."""
        with open(path, "rb") as f:
            self.header = f.read(self.HEADER_SIZE)
            if len(self.header) != self.HEADER_SIZE:
                raise ValueError(
                    f"Invalid BMP header length: expected {self.HEADER_SIZE}, got {len(self.header)}"
                )

            pixel_bytes = f.read(self.PIXEL_COUNT)
            if len(pixel_bytes) != self.PIXEL_COUNT:
                raise ValueError(
                    f"Invalid pixel payload: expected {self.PIXEL_COUNT} bytes, got {len(pixel_bytes)}"
                )

            self.pixels = np.frombuffer(pixel_bytes, dtype=np.uint8).reshape(
                (self.HEIGHT, self.WIDTH)
            )

            # In BMP, positive height means rows are stored bottom-up.
            # For processing/display we keep `self.pixels` top-down.
            h = self._read_bmp_height()
            self._bottom_up = h > 0
            if self._bottom_up:
                self.pixels = np.flipud(self.pixels)

    def write(self, path, pixels=None):
        """
        Save BMP file using stored 1078-byte header and a 256x256 pixel matrix.

        If `pixels` is not provided, this writes `self.pixels`.
        """
        if self.header is None or len(self.header) != self.HEADER_SIZE:
            raise ValueError("Header is missing or invalid. Call read() first.")

        matrix = self.pixels if pixels is None else pixels
        if matrix is None:
            raise ValueError("No pixel matrix available to write.")

        matrix = np.asarray(matrix, dtype=np.uint8)
        if matrix.shape != (self.HEIGHT, self.WIDTH):
            raise ValueError(
                f"Invalid matrix shape: expected ({self.HEIGHT}, {self.WIDTH}), got {matrix.shape}"
            )

        # Convert back to BMP storage order if needed.
        matrix_to_write = np.flipud(matrix) if self._bottom_up else matrix

        with open(path, "wb") as f:
            f.write(self.header)
            f.write(matrix_to_write.tobytes())


def _self_test():
    test_files = ["Lenna256an.bmp", "test_pepper.bmp"]

    for file_name in test_files:
        path = Path(file_name)
        if not path.exists():
            print(f"{file_name}: missing")
            continue

        handler = GrayscaleBMP256()
        handler.read(path)

        out_path = path.with_name(path.stem + "_copy.bmp")
        handler.write(out_path)

        same = path.read_bytes() == out_path.read_bytes()
        print(f"{file_name}: {'OK' if same else 'DIFF'}")


if __name__ == "__main__":
    root = tk.Tk()
    app = CompressionGUI(root)
    root.mainloop()
