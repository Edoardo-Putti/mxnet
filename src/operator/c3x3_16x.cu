/*!
 * Copyright (c) 2017 by Contributors
 * \file c3x3_16x.cu
 * \brief
 * \author Bing Xu, Jun Wu
*/

#include "./c3x3_16x-inl.h"
#include <vector>
#if MXNET_USE_CUDNN == 1
#include "./cudnn_convolution-inl.h"
#endif  // MXNET_USE_CUDNN

namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(c3x3_16Param param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  Operator *op = NULL;
  // If 1D convolution, use MXNet implementation
  if (param.kernel.ndim() == 1) {
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      op = new c3x3_16Op<gpu, DType>(param);
    })
    return op;
  }
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new c3x3_16Op<gpu, DType>(param);
  })
  return op;
}

}  // namespace op
}  // namespace mxnet

