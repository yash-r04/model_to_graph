"""Builds two tiny .tflite files with the `tflite` flatbuffer package only (no TensorFlow needed).

    python examples/make_test_tflite.py

Writes examples/tiny_cnn.tflite (fp32) and examples/tiny_cnn_int8.tflite (int8 weights/activations,
int32 bias). Same topology and the same hand-computed numbers as examples/tiny_cnn.csv:

    input 1x8x8x3 (NHWC)
      CONV_2D           w 4x3x3x3 (OHWI) + bias 4  -> 1x8x8x4
      RELU                                         -> 1x8x8x4
      DEPTHWISE_CONV_2D w 1x3x3x4                  -> 1x8x8x4
      RESHAPE           shape [1, 256]             -> 1x256
      FULLY_CONNECTED   w 10x256 + bias 10         -> 1x10
      SOFTMAX                                      -> 1x10

NOTE: the files are structurally valid (rooftool, and the flatbuffer reader, load them) but the
builtin-options tables (padding, strides, ...) are left out, so they are meant for the rooftool
frontend, not for running in the TFLite interpreter.
"""
import pathlib
import flatbuffers
import numpy as np
import tflite as T

F32, I32, I8 = 0, 2, 9
OPS = T.BuiltinOperator


def _ivec(b, start, items):
    start(b, len(items))
    for x in reversed(items):
        b.PrependInt32(x)
    return b.EndVector()


def _ovec(b, start, offsets):
    start(b, len(offsets))
    for o in reversed(offsets):
        b.PrependUOffsetTRelative(o)
    return b.EndVector()


def build(path, act, wt, bias, batch=1):
    """act/wt/bias: tflite TensorType codes used for activations, weights and biases.
    batch: leading dim of every activation tensor; -1 makes it dynamic (use --dynamic-dim)."""
    rng = np.random.default_rng(0)
    np_of = {F32: np.float32, I32: np.int32, I8: np.int8}

    def data(shape, code):
        return rng.integers(-3, 4, size=shape).astype(np_of[code]).tobytes()

    # (name, shape, type, buffer-bytes or None)
    tensors = [
        ("input",     [1, 8, 8, 3], act,  None),
        ("conv_w",    [4, 3, 3, 3], wt,   data([4, 3, 3, 3], wt)),
        ("conv_b",    [4],          bias, data([4], bias)),
        ("conv_out",  [1, 8, 8, 4], act,  None),
        ("relu_out",  [1, 8, 8, 4], act,  None),
        ("dw_w",      [1, 3, 3, 4], wt,   data([1, 3, 3, 4], wt)),
        ("dw_out",    [1, 8, 8, 4], act,  None),
        ("rs_shape",  [2],          I32,  np.array([1, 256], np.int32).tobytes()),
        ("flat",      [1, 256],     act,  None),
        ("fc_w",      [10, 256],    wt,   data([10, 256], wt)),
        ("fc_b",      [10],         bias, data([10], bias)),
        ("fc_out",    [1, 10],      act,  None),
        ("probs",     [1, 10],      act,  None),
    ]
    tensors = [(n, ([batch] + sh[1:]) if raw is None else sh, t, raw) for n, sh, t, raw in tensors]
    opcodes = [OPS.CONV_2D, OPS.RELU, OPS.DEPTHWISE_CONV_2D, OPS.RESHAPE,
               OPS.FULLY_CONNECTED, OPS.SOFTMAX]
    #            opcode idx, inputs,     outputs
    operators = [(0, [0, 1, 2], [3]), (1, [3], [4]), (2, [4, 5], [6]),
                 (3, [6, 7], [8]), (4, [8, 9, 10], [11]), (5, [11], [12])]

    b = flatbuffers.Builder(4096)

    # buffers: index 0 is the empty buffer by TFLite convention
    buf_offsets, buf_index = [], {}
    T.BufferStart(b); buf_offsets.append(T.BufferEnd(b))
    for i, (_, _, _, raw) in enumerate(tensors):
        if raw is not None:
            vec = b.CreateNumpyVector(np.frombuffer(raw, dtype=np.uint8))
            T.BufferStart(b); T.BufferAddData(b, vec)
            buf_offsets.append(T.BufferEnd(b)); buf_index[i] = len(buf_offsets) - 1
    buffers = _ovec(b, T.ModelStartBuffersVector, buf_offsets)

    codes = []
    for c in opcodes:
        T.OperatorCodeStart(b)
        T.OperatorCodeAddDeprecatedBuiltinCode(b, c)
        T.OperatorCodeAddBuiltinCode(b, c)
        T.OperatorCodeAddVersion(b, 1)
        codes.append(T.OperatorCodeEnd(b))
    codes_v = _ovec(b, T.ModelStartOperatorCodesVector, codes)

    t_offsets = []
    for i, (name, shape, typ, _) in enumerate(tensors):
        nm = b.CreateString(name)
        sh = _ivec(b, T.TensorStartShapeVector, shape)
        T.TensorStart(b); T.TensorAddShape(b, sh); T.TensorAddType(b, typ)
        T.TensorAddBuffer(b, buf_index.get(i, 0)); T.TensorAddName(b, nm)
        t_offsets.append(T.TensorEnd(b))
    tensors_v = _ovec(b, T.SubGraphStartTensorsVector, t_offsets)

    o_offsets = []
    for oc, ins, outs in operators:
        iv = _ivec(b, T.OperatorStartInputsVector, ins)
        ov = _ivec(b, T.OperatorStartOutputsVector, outs)
        T.OperatorStart(b); T.OperatorAddOpcodeIndex(b, oc)
        T.OperatorAddInputs(b, iv); T.OperatorAddOutputs(b, ov)
        o_offsets.append(T.OperatorEnd(b))
    ops_v = _ovec(b, T.SubGraphStartOperatorsVector, o_offsets)

    g_in = _ivec(b, T.SubGraphStartInputsVector, [0])
    g_out = _ivec(b, T.SubGraphStartOutputsVector, [12])
    gname = b.CreateString("main")
    T.SubGraphStart(b)
    T.SubGraphAddTensors(b, tensors_v); T.SubGraphAddInputs(b, g_in)
    T.SubGraphAddOutputs(b, g_out); T.SubGraphAddOperators(b, ops_v); T.SubGraphAddName(b, gname)
    sg = T.SubGraphEnd(b)
    sg_v = _ovec(b, T.ModelStartSubgraphsVector, [sg])

    T.ModelStart(b)
    T.ModelAddVersion(b, 3); T.ModelAddOperatorCodes(b, codes_v)
    T.ModelAddSubgraphs(b, sg_v); T.ModelAddBuffers(b, buffers)
    b.Finish(T.ModelEnd(b), b"TFL3")
    pathlib.Path(path).write_bytes(bytes(b.Output()))
    print("saved", path)


if __name__ == "__main__":
    here = pathlib.Path(__file__).parent
    build(here / "tiny_cnn.tflite", F32, F32, F32)
    build(here / "tiny_cnn_int8.tflite", I8, I8, I32)
