#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace needle {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);


/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
 * here by default.
 */
struct AlignedArray {
  AlignedArray(const size_t size) {
    int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);
    if (ret != 0) throw std::bad_alloc();
    this->size = size;
  }
  ~AlignedArray() { free(ptr); }
  size_t ptr_as_int() {return (size_t)ptr; }
  scalar_t* ptr;
  size_t size;
};



void Fill(AlignedArray* out, scalar_t val) {
  /**
   * Fill the values of an aligned array with val
   */
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}



void Compact(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory
   *
   * Args:
   *   a: non-compact representation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   *
   * Returns:
   *  void (you need to modify out directly, rather than returning anything; this is true for all the
   *  function will implement here, so we won't repeat this note.)
   */
  /// BEGIN SOLUTION
  size_t ndim = shape.size();
  if (ndim == 0) return;
  size_t total = 1;
  for (auto s : shape) total *= static_cast<size_t>(s);
  std::vector<size_t> idx(ndim, 0);
  size_t out_pos = 0;
  for (size_t cnt = 0; cnt < total; ++cnt) {
    size_t in_pos = offset;
    for (size_t d = 0; d < ndim; ++d) {
      in_pos += static_cast<size_t>(strides[d]) * idx[d];
    }
    out->ptr[out_pos++] = a.ptr[in_pos];
    for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
      if (++idx[d] < static_cast<size_t>(shape[d])) break;
      idx[d] = 0;
    }
  }
  /// END SOLUTION
}

void EwiseSetitem(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  size_t ndim = shape.size();
  if (ndim == 0) return;
  size_t total = 1;
  for (auto s : shape) total *= static_cast<size_t>(s);
  std::vector<size_t> idx(ndim, 0);
  size_t in_pos = 0; 
  for (size_t cnt = 0; cnt < total; ++cnt) {
    size_t out_pos = offset;
    for (size_t d = 0; d < ndim; ++d) {
      out_pos += static_cast<size_t>(strides[d]) * idx[d];
    }
    out->ptr[out_pos] = a.ptr[in_pos++];
    for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
      if (++idx[d] < static_cast<size_t>(shape[d])) break;
      idx[d] = 0;
    }
  }
  /// END SOLUTION
}

void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but convenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */

  /// BEGIN SOLUTION
  size_t ndim = shape.size();
  if (ndim == 0) return;
  std::vector<size_t> idx(ndim, 0);
  for (size_t cnt = 0; cnt < size; ++cnt) {
    size_t out_pos = offset;
    for (size_t d = 0; d < ndim; ++d) {
      out_pos += static_cast<size_t>(strides[d]) * idx[d];
    }
    out->ptr[out_pos] = val;
    for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
      if (++idx[d] < static_cast<size_t>(shape[d])) break;
      idx[d] = 0;
    }
  }
  /// END SOLUTION
}

void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + b.ptr[i];
  }
}

void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + val;
  }
}


/**
 * In the code the follows, use the above template to create analogous element-wise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

template <typename F>
inline void ewise_op(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, F f) {
  for (size_t i = 0; i < a.size; ++i) out->ptr[i] = f(a.ptr[i], b.ptr[i]);
}
template <typename F>
inline void scalar_op(const AlignedArray& a, scalar_t val, AlignedArray* out, F f) {
  for (size_t i = 0; i < a.size; ++i) out->ptr[i] = f(a.ptr[i], val);
}
template <typename F>
inline void unary_op(const AlignedArray& a, AlignedArray* out, F f) {
  for (size_t i = 0; i < a.size; ++i) out->ptr[i] = f(a.ptr[i]);
}

void EwiseMul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  ewise_op(a, b, out, [](scalar_t x, scalar_t y) { return x * y; });
}
void ScalarMul(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  scalar_op(a, val, out, [](scalar_t x, scalar_t v) { return x * v; });
}
void EwiseDiv(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  ewise_op(a, b, out, [](scalar_t x, scalar_t y) { return x / y; });
}
void ScalarDiv(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  scalar_op(a, val, out, [](scalar_t x, scalar_t v) { return x / v; });
}
void ScalarPower(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  unary_op(a, out, [val](scalar_t x) { return std::pow(x, val); });
}
void EwiseMaximum(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  ewise_op(a, b, out, [](scalar_t x, scalar_t y) { return x > y ? x : y; });
}
void ScalarMaximum(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  scalar_op(a, val, out, [](scalar_t x, scalar_t v) { return x > v ? x : v; });
}
void EwiseEq(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  ewise_op(a, b, out, [](scalar_t x, scalar_t y) { return static_cast<scalar_t>(x == y); });
}
void ScalarEq(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  scalar_op(a, val, out, [](scalar_t x, scalar_t v) { return static_cast<scalar_t>(x == v); });
}
void EwiseGe(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  ewise_op(a, b, out, [](scalar_t x, scalar_t y) { return static_cast<scalar_t>(x >= y); });
}
void ScalarGe(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  scalar_op(a, val, out, [](scalar_t x, scalar_t v) { return static_cast<scalar_t>(x >= v); });
}
void EwiseLog(const AlignedArray& a, AlignedArray* out) {
  unary_op(a, out, [](scalar_t x) { return std::log(x); });
}
void EwiseExp(const AlignedArray& a, AlignedArray* out) {
  unary_op(a, out, [](scalar_t x) { return std::exp(x); });
}
void EwiseTanh(const AlignedArray& a, AlignedArray* out) {
  unary_op(a, out, [](scalar_t x) { return std::tanh(x); });
}

inline scalar_t apply_fused_op(scalar_t x, int32_t op, scalar_t param) {
  switch (op) {
    case 1:
      return x + param;
    case 2:
      return x * param;
    case 3:
      return x / param;
    case 4:
      return std::pow(x, param);
    case 5:
      return -x;
    case 6:
      return std::exp(x);
    case 7:
      return std::log(x);
    case 8:
      return std::tanh(x);
    case 9:
      return x > 0 ? x : 0;
    default:
      return x;
  }
}

void FusedElementwise(const AlignedArray& base,
                      const std::vector<AlignedArray*>& outs,
                      const std::vector<int32_t>& op_codes,
                      const std::vector<float>& op_params) {
  size_t n_ops = op_codes.size();
  if (n_ops == 0) return;
  if (outs.size() != n_ops || op_params.size() != n_ops)
    throw std::runtime_error("EW-Fuse metadata mismatch.");
  for (size_t idx = 0; idx < base.size; ++idx) {
    scalar_t val = base.ptr[idx];
    for (size_t op_idx = 0; op_idx < n_ops; ++op_idx) {
      val = apply_fused_op(val, op_codes[op_idx], op_params[op_idx]);
      outs[op_idx]->ptr[idx] = val;
    }
  }
}


void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m, uint32_t n,
            uint32_t p) {
  /**
   * Multiply two (compact) matrices into an output (also compact) matrix.  For this implementation
   * you can use the "naive" three-loop algorithm.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: compact 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   */

  /// BEGIN SOLUTION
  for (size_t i = 0; i < out->size; ++i) out->ptr[i] = 0.0f;
  const scalar_t* A = a.ptr;
  const scalar_t* B = b.ptr;
  scalar_t* C = out->ptr;
  for (uint32_t i = 0; i < m; ++i) {
    for (uint32_t k = 0; k < n; ++k) {
      scalar_t aik = A[i * n + k];
      const scalar_t* b_row = &B[k * p];
      scalar_t* c_row = &C[i * p];
      for (uint32_t j = 0; j < p; ++j) {
        c_row[j] += aik * b_row[j];
      }
    }
  }
  /// END SOLUTION
}

inline void AlignedDot(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ out) {

  /**
   * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
   * the result to the existing out, which you should not set to zero beforehand).  We are including
   * the compiler flags here that enable the compile to properly use vector operators to implement
   * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and
   * out don't have any overlapping memory (which is necessary in order for vector operations to be
   * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b,
   * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the
   * compiler that the input array will be aligned to the appropriate blocks in memory, which also
   * helps the compiler vectorize the code.
   *
   * Args:
   *   a: compact 2D array of size TILE x TILE
   *   b: compact 2D array of size TILE x TILE
   *   out: compact 2D array of size TILE x TILE to write to
   */

  a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
  b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
  out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

  /// BEGIN SOLUTION
  for (int i = 0; i < TILE; ++i) {
    float* out_row = out + i * TILE;
    for (int k = 0; k < TILE; ++k) {
      float aik = a[i * TILE + k];
      const float* b_row = b + k * TILE;
      for (int j = 0; j < TILE; ++j) {
        out_row[j] += aik * b_row[j];
      }
    }
  }
  /// END SOLUTION
}

void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m,
                 uint32_t n, uint32_t p) {
  /**
   * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
   * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
   *   a[m/TILE][n/TILE][TILE][TILE]
   * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
   * function should call `AlignedDot()` implemented above).
   *
   * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
   * assume that this division happens without any remainder.
   *
   * Args:
   *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
   *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
   *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   *
   */
  /// BEGIN SOLUTION
  const uint32_t mT = m / TILE;
  const uint32_t nT = n / TILE;
  const uint32_t pT = p / TILE;
  for (size_t t = 0; t < out->size; ++t) out->ptr[t] = 0.0f;
  const float* A = a.ptr;
  const float* B = b.ptr;
  float* C = out->ptr;
  for (uint32_t ti = 0; ti < mT; ++ti) {
    for (uint32_t tj = 0; tj < pT; ++tj) {
      size_t base_c = ((size_t)ti * pT + tj) * (size_t)TILE * (size_t)TILE;
      float* c_tile = C + base_c;
      for (uint32_t tk = 0; tk < nT; ++tk) {
        size_t base_a = ((size_t)ti * nT + tk) * (size_t)TILE * (size_t)TILE;
        size_t base_b = ((size_t)tk * pT + tj) * (size_t)TILE * (size_t)TILE;
        const float* a_tile = A + base_a;
        const float* b_tile = B + base_b;
        AlignedDot(a_tile, b_tile, c_tile);
      }
    }
  }
  /// END SOLUTION
}

void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
  const size_t n_out = out->size;
  const scalar_t* ap = a.ptr;
  scalar_t* op = out->ptr;
  for (size_t i = 0; i < n_out; ++i) {
    size_t base = i * reduce_size;
    scalar_t m = -std::numeric_limits<scalar_t>::infinity();
    for (size_t j = 0; j < reduce_size; ++j) {
      scalar_t v = ap[base + j];
      if (v > m) m = v;
    }
    op[i] = m;
  }
  /// END SOLUTION
}

void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
  const size_t n_out = out->size;
  const scalar_t* ap = a.ptr;
  scalar_t* op = out->ptr;
  for (size_t i = 0; i < n_out; ++i) {
    size_t base = i * reduce_size;
    scalar_t s = 0.0f;
    for (size_t j = 0; j < reduce_size; ++j) {
      s += ap[base + j];
    }
    op[i] = s;
  }
  /// END SOLUTION
}

}  // namespace cpu
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);
  m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
  m.def("fused_elementwise", FusedElementwise);
}
