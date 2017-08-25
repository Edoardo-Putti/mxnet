/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2017 Microsoft
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file psroi_pooling.cc
 * \brief psroi pooling operator
 * \author Yi Li, Tairui Chen, Guodong Zhang, Haozhi Qi, Jifeng Dai
*/
#include "./psroi_pooling-inl.h"
#include <mshadow/base.h>
#include <mshadow/tensor.h>
#include <mshadow/packet-inl.h>
#include <mshadow/dot_engine-inl.h>
#include <cassert>

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace mshadow {
template<typename DType>
inline void PSROIPoolForward(const Tensor<cpu, 4, DType> &out,
                           const Tensor<cpu, 4, DType> &data,
                           const Tensor<cpu, 2, DType> &bbox,
                           const float spatial_scale_,
                           const int output_dim_,
                           const int group_size_) {
  // NOT_IMPLEMENTED;
  return;
}

template<typename DType>
inline void PSROIPoolBackwardAcc(const Tensor<cpu, 4, DType> &in_grad,
                            const Tensor<cpu, 4, DType> &out_grad,
                            const Tensor<cpu, 2, DType> &bbox,
                            const float spatial_scale_,
                            const int output_dim_,
                            const int group_size_) {
  // NOT_IMPLEMENTED;
  return;
}
}  // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(PSROIPoolingParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new PSROIPoolingOp<cpu, DType>(param);
  });
  return op;
}

Operator *PSROIPoolingProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                           std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(PSROIPoolingParam);

MXNET_REGISTER_OP_PROPERTY(_contrib_PSROIPooling, PSROIPoolingProp)
.describe("Performs region-of-interest pooling on inputs. Resize bounding box coordinates by "
"spatial_scale and crop input feature maps accordingly. The cropped feature maps are pooled "
"by max pooling to a fixed size output indicated by pooled_size. batch_size will change to "
"the number of region bounding boxes after PSROIPooling")
.add_argument("data", "Symbol", "Input data to the pooling operator, a 4D Feature maps")
.add_argument("rois", "Symbol", "Bounding box coordinates, a 2D array of "
"[[batch_index, x1, y1, x2, y2]]. (x1, y1) and (x2, y2) are top left and down right corners "
"of designated region of interest. batch_index indicates the index of corresponding image "
"in the input data")
.add_arguments(PSROIPoolingParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
