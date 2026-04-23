import heapq
from collections import defaultdict


class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def build_frequency_table(text):
    freq = defaultdict(int)
    for ch in text:
        freq[ch] += 1
    return freq


def build_huffman_tree(freq_table):
    heap = [Node(ch, freq) for ch, freq in freq_table.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0] if heap else None


def build_codes(node, prefix="", codes=None):
    if codes is None:
        codes = {}
    if node is None:
        return codes
    if node.char is not None:
        codes[node.char] = prefix if prefix else "0"
    else:
        build_codes(node.left, prefix + "0", codes)
        build_codes(node.right, prefix + "1", codes)
    return codes


def encode(text):
    if not text:
        return "", {}

    freq_table = build_frequency_table(text)
    tree = build_huffman_tree(freq_table)
    codes = build_codes(tree)

    encoded = "".join(codes[ch] for ch in text)
    return encoded, codes


def decode(encoded_text, codes):
    if not encoded_text or not codes:
        return ""

    reverse_codes = {v: k for k, v in codes.items()}
    current = []
    decoded = []

    for bit in encoded_text:
        current.append(bit)
        if "".join(current) in reverse_codes:
            decoded.append(reverse_codes["".join(current)])
            current = []

    return "".join(decoded)


def compression_ratio(original, encoded):
    original_bits = len(original) * 8
    encoded_bits = len(encoded)
    if original_bits == 0:
        return 0.0
    return 1.0 - encoded_bits / original_bits


if __name__ == "__main__":
    text = "hello huffman encoding"
    print(f"Original text: {text}")

    encoded, codes = encode(text)
    print(f"Encoded bits:  {encoded}")
    print(f"Codes table:   {codes}")

    decoded = decode(encoded, codes)
    print(f"Decoded text:  {decoded}")

    ratio = compression_ratio(text, encoded)
    print(f"Compression ratio: {ratio:.2%}")
