# Safetensors Format Description

- 8 bytes: `N`, an unsigned little-endian 64-bit integer, containing the size of the header
- N bytes: a JSON UTF-8 string representing the header.
  - The header data MUST begin with a `{` character (0x7B).
  - The header data MAY be trailing padded with whitespace (0x20).
  - The header is a dict like `{"TENSOR_NAME": {"dtype": "F16", "shape": [1, 16, 256], "data_offsets": [BEGIN, END]}, "NEXT_TENSOR_NAME": {...}, ...}`,
    - `data_offsets` point to the tensor data relative to the beginning of the byte buffer (i.e. not an absolute position in the file),
      with `BEGIN` as the starting offset and `END` as the one-past offset (so total tensor byte size = `END - BEGIN`).
  - A special key `__metadata__` is allowed to contain free form string-to-string map. Arbitrary JSON is not allowed, all values must be strings.
- Rest of the file: byte-buffer.

Notes:
 - Duplicate keys are disallowed. Not all parsers may respect this.
 - In general the subset of JSON is implicitly decided by `serde_json` for this library. Anything obscure might be modified at a later time, that odd ways to represent integer, newlines and escapes in utf-8 strings. This would only be done for safety concerns
 - Tensor values are not checked against, in particular NaN and +/-Inf could be in the file
 - Empty tensors (tensors with 1 dimension being 0) are allowed. They are not storing any data in the databuffer, yet retaining size in the header. They don't really bring a lot of values but are accepted since they are valid tensors from traditional tensor libraries perspective (torch, tensorflow, numpy, ..).
 - 0-rank Tensors (tensors with shape `[]`) are allowed, they are merely a scalar.
 - The byte buffer needs to be entirely indexed, and cannot contain holes. This prevents the creation of polyglot files.
 - Endianness: Little-endian.
 - Order: 'C' or row-major.
 - Notes: Some smaller than 1 byte dtypes appeared, which make alignment tricky. Non traditional APIs might be required for those.
