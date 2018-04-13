https://transcranial.github.io/keras-js-docs/conversion/
to encode a model run the following command from this directory:

py encoder.py <PATH_TO_MODEL>

"The quantize flag enables weights-wise 8-bit uint quantization from 32-bit float,
using a simple linear min/max scale calculated for every layer weights matrix.
This will result in a roughly 4x reduction in the model file size. For example,
the model file for Inception-V3 is reduced from 92 MB to 23 MB. Client-side,
Keras.js then restores uint8-quantized weights back to float32 during model initialization."

py encoder.py -q <PATH_TO_MODEL>