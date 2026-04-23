import heapq
from collections import Counter


class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(text):
    frequency = Counter(text)
    heap = [HuffmanNode(char, freq) for char, freq in frequency.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq)
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

    root = build_huffman_tree(text)
    codes = build_codes(root)
    encoded = "".join(codes[char] for char in text)
    return encoded, codes


def decode(encoded_text, codes):
    if not encoded_text or not codes:
        return ""

    reverse_codes = {v: k for k, v in codes.items()}
    current = ""
    decoded = []

    for bit in encoded_text:
        current += bit
        if current in reverse_codes:
            decoded.append(reverse_codes[current])
            current = ""

    return "".join(decoded)


def compression_ratio(original, encoded):
    """Calculate compression ratio.

    Args:
        original: original string (each character assumed to be 8 bits / 1 byte).
        encoded: bit-string of '0'/'1' characters where len(encoded) equals the
                 number of encoded bits (not bytes), allowing a direct comparison
                 with the original bit count.
    Returns:
        Float in [0, 1) representing the fraction of bits saved (1.0 = perfect).
    """
    original_bits = len(original) * 8
    encoded_bits = len(encoded)
    if original_bits == 0:
        return 0.0
    return 1 - (encoded_bits / original_bits)


if __name__ == "__main__":
    sample_text = "aceasta este un exemplu de compresie Huffman"

    print("Text original:", sample_text)

    encoded_text, huffman_codes = encode(sample_text)
    print("\nCoduri Huffman:")
    for char, code in sorted(huffman_codes.items()):
        display_char = repr(char) if char == " " else char
        print(f"  '{display_char}': {code}")

    print("\nText codificat:", encoded_text)
    print("Lungime originala (biti):", len(sample_text) * 8)
    print("Lungime codificata (biti):", len(encoded_text))

    ratio = compression_ratio(sample_text, encoded_text)
    print(f"Rata de compresie: {ratio:.2%}")

    decoded_text = decode(encoded_text, huffman_codes)
    print("\nText decodificat:", decoded_text)
    print("Decodificare corecta:", sample_text == decoded_text)
