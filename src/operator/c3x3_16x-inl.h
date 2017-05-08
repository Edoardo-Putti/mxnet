/*!
 * Copyright (c) 2017 by Contributors
 * \file c3x3_16-inl.h
 * \brief
 * \ref: https://github.com/Yangqing/caffe/wiki/c3x3_16-in-Caffe:-a-memo
 * \author Bing Xu, Jun Wu
*/
#ifndef MXNET_OPERATOR_c3x3_16_INL_H_
#define MXNET_OPERATOR_c3x3_16_INL_H_

#include <mxnet/io.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <dmlc/logging.h>
#include <dmlc/optional.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./nn/im2col.h"


namespace mxnet {
namespace op {

namespace c3x3_16 {
enum c3x3_16OpInputs {kData, kWeight, kBias, kExtra, kExtra_Bias};
enum c3x3_16OpOutputs {kOut};
enum c3x3_16OpResource {kTempSpace};
enum c3x3_16OpCudnnTune {kOff, kLimited, kFastest};
}

struct c3x3_16Param : public dmlc::Parameter<c3x3_16Param> {
  TShape kernel;
  TShape stride;
  TShape dilate;
  TShape pad;
  uint32_t num_filter;
  uint32_t num_group;
  uint64_t workspace;
  bool no_bias;
  dmlc::optional<int> cudnn_tune;
  bool cudnn_off;
  dmlc::optional<int> layout;
  DMLC_DECLARE_PARAMETER(c3x3_16Param) {
    DMLC_DECLARE_FIELD(kernel).describe("c3x3_16 kernel size: (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(stride).set_default(TShape())
    .describe("c3x3_16 stride: (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(dilate).set_default(TShape())
    .describe("c3x3_16 dilate: (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(pad).set_default(TShape())
    .describe("pad for c3x3_16: (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(num_filter).set_range(1, 100000)
    .describe("c3x3_16 filter(channel) number");
    DMLC_DECLARE_FIELD(num_group).set_default(1)
    .describe("Number of group partitions.");
    DMLC_DECLARE_FIELD(workspace).set_default(1024).set_range(0, 8192)
    .describe("Maximum temperal workspace allowed for c3x3_16 (MB).");
    DMLC_DECLARE_FIELD(no_bias).set_default(false)
    .describe("Whether to disable bias parameter.");
    DMLC_DECLARE_FIELD(cudnn_tune)
    .add_enum("off", c3x3_16::kOff)
    .add_enum("limited_workspace", c3x3_16::kLimited)
    .add_enum("fastest", c3x3_16::kFastest)
    .set_default(dmlc::optional<int>())
        .describe("Whether to pick c3x3_16 algo by running performance test.");
    DMLC_DECLARE_FIELD(cudnn_off).set_default(false)
    .describe("Turn off cudnn for this layer.");
    DMLC_DECLARE_FIELD(layout)
    .add_enum("NCW", mshadow::kNCW)
    .add_enum("NCHW", mshadow::kNCHW)
    .add_enum("NCDHW", mshadow::kNCDHW)
    .add_enum("NHWC", mshadow::kNHWC)
    .add_enum("NDHWC", mshadow::kNDHWC)
    .set_default(dmlc::optional<int>())
    .describe("Set layout for input, output and weight. Empty for\n    "
              "default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.");
  }
};

template<typename xpu, typename DType>
class c3x3_16Op : public Operator {
 public:
  explicit c3x3_16Op(c3x3_16Param p) {
    this->param_ = p;
    // convert MBytes first to Bytes and then to elements.
    param_.workspace = (param_.workspace << 20) / sizeof(DType);
    CHECK(param_.layout.value() == mshadow::kNCW ||
          param_.layout.value() == mshadow::kNCHW ||
          param_.layout.value() == mshadow::kNCDHW)
      << "Only support NCW, NCHW and NCDHW layout";
  }

  void debug_shape(const std::vector<TBlob> &data){

    for (int i = 0; i < data.size(); i++) {
        TBlob tb = data[i];
        auto ndim = tb.shape_.ndim();
        LOG(INFO) << "#########" << ndim;
        for (int j = 0; j < ndim; j++) {
            LOG(INFO) << "^^^^^^^" << tb.shape_.data_stack_[j];
        }
    }

  }



  void expand(const std::vector<TBlob> &data, const OpContext &ctx) {

    using namespace mshadow;
    using namespace mshadow::expr;
    TBlob otb = data[c3x3_16::kWeight];
    int on = otb.shape_.data_stack_[0];
    int oc = otb.shape_.data_stack_[1];
    int oh = otb.shape_.data_stack_[2];
    int ow = otb.shape_.data_stack_[3];

    Stream<xpu>* s = ctx.get_stream<xpu>();

    //kernel_size = oh * ow;
    Tensor<xpu, 3, DType> x_weight_origin = data[c3x3_16::kWeight].get_with_shape<xpu, 3, DType>(
    						Shape3(on, oc, oh * ow), s);

    TBlob etb = data[c3x3_16::kExtra];
    int en = etb.shape_.data_stack_[0];
    int ec = etb.shape_.data_stack_[1];
    int eh = etb.shape_.data_stack_[2];
    int ew = etb.shape_.data_stack_[3];

    Tensor<xpu, 3, DType> x_weight_expand = data[c3x3_16::kExtra].get_with_shape<xpu, 3, DType>(
    						Shape3(en, ec, eh * ew), s);

    Tensor<cpu, 3, DType> weight_origin = NewTensor<cpu, DType>(Shape3(on, oc, oh * ow), 0.0f);
    Tensor<cpu, 3, DType> weight_expand = NewTensor<cpu, DType>(Shape3(en, ec, eh * ew), 0.0f);

    Copy(weight_origin, x_weight_origin, s);
    Copy(weight_expand, x_weight_expand, s);
    weight_expand = 0;

    int expand_num = en / on;
    for (int i = 0; i < weight_origin.size(0); i++) {
    	for (int j = 0; j < weight_origin.size(1); j++) {
           for(int k = 0; k < weight_origin.size(2); k++) {
			   //LOG(INFO) << "i " << i << " j " << j << " k " << k;

               for (int w = i * expand_num, t = k; w < i * expand_num + expand_num; w += 2, t++) {
			   	  //LOG(INFO) << "w " << w << " j " << j << " t " << t << "expand_num" << expand_num;
                  weight_expand[w][j][t] = weight_origin[i][j][k];

				  if (t == 3 || t == 4 || t == 5) {
			     	w = w - 1;
					continue;
                  }
				  if (t > 8) {
					t = 0;
				  }
    			  switch(t) {
                   	case 0:
                   	    weight_expand[w+1][j][6] = weight_origin[i][j][k];
                   	    break;
                   	case 1:
                   	    weight_expand[w+1][j][7] = weight_origin[i][j][k];
                   	    break;
                   	case 2:
                   	    weight_expand[w+1][j][8] = weight_origin[i][j][k];
                   	    break;
                   	case 6:
                   	    weight_expand[w+1][j][0] = weight_origin[i][j][k];
                   	    break;
                   	case 7:
                   	    weight_expand[w+1][j][1] = weight_origin[i][j][k];
                   	    break;
                   	case 8:
                   	    weight_expand[w+1][j][2] = weight_origin[i][j][k];
                   	    break;
                   	default:
                   	    break;

                  }


               }
           }
       }
   	 }


    Copy(x_weight_origin, weight_origin, s);
    Copy(x_weight_expand, weight_expand, s);

    FreeSpace(&weight_origin);
    FreeSpace(&weight_expand);

    TBlob obb = data[c3x3_16::kBias];
    on = obb.shape_.data_stack_[0];

    Tensor<xpu, 1, DType> x_bias_origin = data[c3x3_16::kBias].get_with_shape<xpu, 1, DType>(
    						Shape1(on), s);


    TBlob ebb = data[c3x3_16::kExtra_Bias];
    en = ebb.shape_.data_stack_[0];

    Tensor<xpu, 1, DType> x_bias_expand = data[c3x3_16::kExtra_Bias].get_with_shape<xpu, 1, DType>(
    						Shape1(en), s);


    Tensor<cpu, 1, DType> bias_origin = NewTensor<cpu, DType>(Shape1(on), 0.0f);
    Tensor<cpu, 1, DType> bias_expand = NewTensor<cpu, DType>(Shape1(en), 0.0f);

    Copy(bias_origin, x_bias_origin, s);
    Copy(bias_expand, x_bias_expand, s);

    for (int i = 0; i < bias_origin.size(0); i++) {
        for (int j = i * expand_num ; j < expand_num + i * expand_num; j++) {
            //LOG(INFO) <<"expand bias" << j;
            bias_expand[j] = bias_origin[i];
        }
    }

    Copy(x_bias_origin, bias_origin, s);
    Copy(x_bias_expand, bias_expand, s);

    FreeSpace(&bias_origin);
    FreeSpace(&bias_expand);
   }


  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(req[c3x3_16::kOut], kWriteTo);
    size_t expected = param_.no_bias ? 3 : 5;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1U);
    CHECK_EQ(req[c3x3_16::kOut], kWriteTo);

    //LOG(INFO) << "in_data";
    //debug_shape(in_data);
    //LOG(INFO) << "out_data";
    //debug_shape(out_data);

	//LOG(INFO) << "----------before expand";
	expand(in_data, ctx);
	//LOG(INFO) << "----------after expand";

    LayerSetUp(in_data[c3x3_16::kData].shape_, out_data[c3x3_16::kOut].shape_);
    Stream<xpu>* s = ctx.get_stream<xpu>();
    // allocate workspace for col_buffer
    Tensor<xpu, 1, DType> workspace = ctx.requested[c3x3_16::kTempSpace]
      .get_space_typed<xpu, 1, DType>(Shape1(col_buffer_size_), s);

    Tensor<xpu, 3, DType> weight_expandd_weight = ctx.requested[c3x3_16::kTempSpace]
      .get_space_typed<xpu, 3, DType>(Shape3(1, conv_out_channels_, kernel_dim_), s);

    // calculate the shape of col_buffer
    TShape col_buffer_shape(num_spatial_axes_ + 1);

    //buffer for 1 pixels
    col_buffer_shape[0] = conv_in_channels_ * param_.kernel.Size();
    for (index_t i = 1; i < col_buffer_shape.ndim(); ++i) {
      col_buffer_shape[i] = out_data[0].shape_[i+1];
    }
    // create a column buffer using workspace and col_buffer_shape
    TBlob col_buffer(workspace.dptr_, col_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);

    // initialize weight and col_buffer 3D tensors for using gemm
    index_t M = conv_out_channels_ / group_;
    index_t N = conv_out_spatial_dim_;
    index_t K = kernel_dim_;
    Tensor<xpu, 3, DType> weight_3d = in_data[c3x3_16::kExtra].get_with_shape<xpu, 3, DType>(
      Shape3(group_, M, K), s);

    //LOG(INFO) <<"weight 3d filters" << weight_3d.shape_[1];

    Tensor<xpu, 3, DType> col_buffer_3d = col_buffer.get_with_shape<xpu, 3, DType>(
      Shape3(group_, K, N), s);
    Tensor<xpu, 4, DType> output_4d = out_data[c3x3_16::kOut].get_with_shape<xpu, 4, DType>(
      Shape4(num_, group_, M, N), s);

	//LOG(INFO) << "befor dot ";
    for (index_t n = 0; n < num_; ++n) {
      // transform image to col_buffer in order to use gemm
      im2col(s, in_data[c3x3_16::kData].dptr<DType>()+n*input_dim_, in_data[c3x3_16::kData].shape_,
             col_buffer.shape_, param_.kernel, param_.pad, param_.stride, param_.dilate,
             col_buffer.dptr<DType>());
      Tensor<xpu, 3, DType> output_3d = output_4d[n];
      for (index_t g = 0; g < group_; ++g) {
        ASSIGN_DISPATCH(output_3d[g], req[c3x3_16::kOut], dot(weight_3d[g], col_buffer_3d[g]));
      }
    }
	//LOG(INFO) << "after dot ";
    if (bias_term_) {
      Tensor<xpu, 1, DType> bias = in_data[c3x3_16::kExtra_Bias].get<xpu, 1, DType>(s);
      Tensor<xpu, 3, DType> output_3d = out_data[c3x3_16::kOut].get_with_shape<xpu, 3, DType>(
        Shape3(num_, conv_out_channels_, conv_out_spatial_dim_), s);
      // has bias term, broadcast it to the same shape of output_3d in channel dim
      output_3d += mshadow::expr::broadcast<1>(bias, output_3d.shape_);
    }
	//LOG(INFO) << "return from forward ";
  }

  void shrink(const std::vector<TBlob> &data, const OpContext &ctx) {

    using namespace mshadow;
    using namespace mshadow::expr;
    TBlob otb = data[c3x3_16::kWeight];
    int on = otb.shape_.data_stack_[0];
    int oc = otb.shape_.data_stack_[1];
    int oh = otb.shape_.data_stack_[2];
    int ow = otb.shape_.data_stack_[3];

    Stream<xpu>* s = ctx.get_stream<xpu>();

    //kernel_size = oh * ow;
    Tensor<xpu, 3, DType> x_weight_origin = data[c3x3_16::kWeight].get_with_shape<xpu, 3, DType>(
    						Shape3(on, oc, oh * ow), s);


    TBlob etb = data[c3x3_16::kExtra];
    int en = etb.shape_.data_stack_[0];
    int ec = etb.shape_.data_stack_[1];
    int eh = etb.shape_.data_stack_[2];
    int ew = etb.shape_.data_stack_[3];

    Tensor<xpu, 3, DType> x_weight_expand = data[c3x3_16::kExtra].get_with_shape<xpu, 3, DType>(
    						Shape3(en, ec, eh * ew), s);

    //LOG(INFO) << "shark------------------------before new";
    //LOG(INFO) << "on " << on << " oc " << oc << " oh * ow " << oh * ow;
    //LOG(INFO) << "en " << en << " ec " << ec << " oh * ow " << eh * ew;
    Tensor<cpu, 3, DType> weight_origin = NewTensor<cpu, DType>(Shape3(on, oc, oh * ow), 0.0f);
    Tensor<cpu, 3, DType> weight_expand = NewTensor<cpu, DType>(Shape3(en, ec, eh * ew), 0.0f);

    Copy(weight_origin, x_weight_origin, s);
    Copy(weight_expand, x_weight_expand, s);

    //LOG(INFO) << "shark------------------------after copy";

    int expand_num = en / on;
    for (int i = 0; i < weight_origin.size(0); i++) {
    	for (int j = 0; j < weight_origin.size(1); j++) {
           for(int k = 0; k < weight_origin.size(2); k++) {
               for (int w = i * expand_num, t = k; w < i * expand_num + expand_num; w += 2, t++) {
                  //weight_expand[w][j][t] = weight_origin[i][j][k];
                  weight_origin[i][j][k] = weight_expand[w][j][t];

				  if (t == 3 || t == 4 || t == 5) {
			     	w = w - 1;
					continue;
                  }
				  if (t > 8) {
					t = 0;
				  }
    			  switch(t) {
                   	case 0:
                        weight_origin[i][j][k] += weight_expand[w+1][j][6];
                   	    break;
                   	case 1:
                   	    weight_origin[i][j][k] += weight_expand[w+1][j][7];
                   	    break;
                   	case 2:
                   	    weight_origin[i][j][k] += weight_expand[w+1][j][8];
                   	    break;
                   	case 6:
                   	    weight_origin[i][j][k] = weight_expand[w+1][j][0];
                   	    break;
                   	case 7:
                   	    weight_origin[i][j][k] = weight_expand[w+1][j][1];
                   	    break;
                   	case 8:
                   	    weight_origin[i][j][k] = weight_expand[w+1][j][2];
                   	    break;
                   	default:
                   	    break;

                  }


               }
           }
       }
   	 }


    Copy(x_weight_origin, weight_origin, s);
    Copy(x_weight_expand, weight_expand, s);

    FreeSpace(&weight_origin);
    FreeSpace(&weight_expand);

    TBlob obb = data[c3x3_16::kBias];
    on = obb.shape_.data_stack_[0];

    Tensor<xpu, 1, DType> x_bias_origin = data[c3x3_16::kBias].get_with_shape<xpu, 1, DType>(
    						Shape1(on), s);

    TBlob ebb = data[c3x3_16::kExtra_Bias];
    en = ebb.shape_.data_stack_[0];

    Tensor<xpu, 1, DType> x_bias_expand = data[c3x3_16::kExtra_Bias].get_with_shape<xpu, 1, DType>(
    						Shape1(en), s);

    Tensor<cpu, 1, DType> bias_origin = NewTensor<cpu, DType>(Shape1(on), 0.0f);
    Tensor<cpu, 1, DType> bias_expand = NewTensor<cpu, DType>(Shape1(en), 0.0f);

    Copy(bias_origin, x_bias_origin, s);
    Copy(bias_expand, x_bias_expand, s);

    int temp_sum = 0;
    for (int i = 0; i < bias_origin.size(0); i++) {
        temp_sum = 0;
        for (int j = i * expand_num ; j < expand_num + i * expand_num; j++) {
            //LOG(INFO) <<"expand bias" << j;
            //bias_expand[j] = bias_origin[i];
            temp_sum += bias_expand[j];
        }
        bias_origin[i] = temp_sum / expand_num;
    }

    Copy(x_bias_origin, bias_origin, s);
    Copy(x_bias_expand, bias_expand, s);

    FreeSpace(&bias_origin);
    FreeSpace(&bias_expand);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob>& out_grad,
                        const std::vector<TBlob>& in_data,
                        const std::vector<TBlob>& out_data,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& in_grad,
                        const std::vector<TBlob>& aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1U);
    size_t expected = param_.no_bias == 0 ? 5 : 3;
    CHECK(in_data.size() == expected && in_grad.size() == expected);
    CHECK_EQ(req.size(), expected);
    CHECK_EQ(in_data[c3x3_16::kWeight].CheckContiguous(), true);

    //LOG(INFO) << "---------------------------------in backward";
    //LOG(INFO) << "in_data";
    //debug_shape(in_data);
    //LOG(INFO) << "out_data";
    //debug_shape(out_data);
    //LOG(INFO) << "in_grad";
    //debug_shape(in_grad);
    //LOG(INFO) << "out_grad";
    //debug_shape(out_grad);

    LayerSetUp(in_grad[c3x3_16::kData].shape_, out_grad[c3x3_16::kOut].shape_);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    // allocate workspace for col_buffer
    Tensor<xpu, 1, DType> workspace = ctx.requested[c3x3_16::kTempSpace]
      .get_space_typed<xpu, 1, DType>(Shape1(col_buffer_size_), s);
    // calculate the shape of col_buffer
	//LOG(INFO) << "col_buffer_shape";
    TShape col_buffer_shape(num_spatial_axes_ + 1);
	//LOG(INFO) << "##########" << col_buffer_shape.ndim();
	//LOG(INFO) << "^^^^^^^" << num_spatial_axes_ + 1;
    col_buffer_shape[0] = conv_in_channels_ * param_.kernel.Size();
    for (index_t i = 1; i < col_buffer_shape.ndim(); ++i) {
      col_buffer_shape[i] = out_grad[c3x3_16::kData].shape_[i+1];
	  //LOG(INFO) << "^^^^^^^" << out_grad[c3x3_16::kData].shape_[i+1];
    }
    // create a column buffer using workspace and col_buffer_shape
    TBlob col_buffer(workspace.dptr_, col_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);

    // initialize weight and col_buffer 3D tensors for using gemm
    // For computing dLoss/d(in_data[kData])
    index_t M = kernel_dim_;
    index_t N = conv_out_spatial_dim_;
    index_t K = conv_out_channels_ / group_;

    Tensor<xpu, 3, DType> weight_3d = in_data[c3x3_16::kExtra].get_with_shape<xpu, 3, DType>(
      Shape3(group_, K, M), s);
    Tensor<xpu, 4, DType> out_grad_4d = out_grad[c3x3_16::kOut].get_with_shape<xpu, 4, DType>(
      Shape4(num_, group_, K, N), s);
    Tensor<xpu, 3, DType> col_buffer_3d = col_buffer.get_with_shape<xpu, 3, DType>(
      Shape3(group_, M, N), s);

    // For computing dLoss/dWeight
    Tensor<xpu, 3, DType> dweight_3d = in_grad[c3x3_16::kExtra].get_with_shape<xpu, 3, DType>(
      Shape3(group_, K, M), s);

    for (index_t n = 0; n < num_; ++n) {
      Tensor<xpu, 3, DType> out_grad_3d = out_grad_4d[n];
      // gradient w.r.t. input data
      for (index_t g = 0; g < group_; ++g) {
        col_buffer_3d[g] = dot(weight_3d[g].T(), out_grad_3d[g]);
      }
      col2im(s, col_buffer.dptr<DType>(), in_grad[c3x3_16::kData].shape_, col_buffer.shape_,
             param_.kernel, param_.pad, param_.stride, param_.dilate,
             in_grad[c3x3_16::kData].dptr<DType>()+n*input_dim_, req[c3x3_16::kData]);

      // gradient w.r.t. weight, dWeight should accumulate across the batch and group
      im2col(s, in_data[c3x3_16::kData].dptr<DType>()+n*input_dim_, in_data[c3x3_16::kData].shape_,
             col_buffer.shape_, param_.kernel, param_.pad, param_.stride, param_.dilate,
             col_buffer.dptr<DType>());
      for (index_t g = 0; g < group_; ++g) {
        if (0 == n) {
          ASSIGN_DISPATCH(dweight_3d[g], req[c3x3_16::kExtra],
                          dot(out_grad_3d[g], col_buffer_3d[g].T()));
        } else {
          dweight_3d[g] += dot(out_grad_3d[g], col_buffer_3d[g].T());
        }
      }

    }

    // gradient w.r.t bias
    if (bias_term_) {
      Tensor<xpu, 1, DType> dbias = in_grad[c3x3_16::kExtra_Bias].get<xpu, 1, DType>(s);
      Tensor<xpu, 3, DType> dout = out_grad[c3x3_16::kOut].get_with_shape<xpu, 3, DType>(
          Shape3(num_, conv_out_channels_, conv_out_spatial_dim_), s);
      ASSIGN_DISPATCH(dbias, req[c3x3_16::kExtra_Bias], sumall_except_dim<1>(dout));
    }

    shrink(in_grad, ctx);
  }

 private:
  void LayerSetUp(const TShape& ishape, const TShape& oshape) {
    channel_axis_ = 1;  // hard code channel axis
    const index_t first_spatial_axis = channel_axis_ + 1;
    const index_t num_axes = param_.kernel.ndim() + 2;
    num_spatial_axes_ = num_axes - first_spatial_axis;
    is_1x1_ = true;
    for (index_t i = 0; i < param_.kernel.ndim(); ++i) {
      is_1x1_ &= param_.kernel[i] == 1 && param_.stride[i] == 1 && param_.pad[i] == 0;
      if (!is_1x1_) break;
    }

    // batch size
    num_ = ishape[0];
    // number of input channels
    channels_ = ishape[1];
    group_ = param_.num_group;

    //expand here
    conv_out_channels_ = param_.num_filter * 15;
    conv_in_channels_ = channels_;
    bias_term_ = !param_.no_bias;
    kernel_dim_ = conv_in_channels_ / group_ * param_.kernel.Size();
    weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;

    conv_out_spatial_dim_ = oshape.ProdShape(2, oshape.ndim());
    col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
    output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
    // size of the column buffer used for storing im2col-ed pixels
    col_buffer_size_ = kernel_dim_ * group_ * conv_out_spatial_dim_;

    // input/output image size (#channels * height * width)
    input_dim_ = ishape.ProdShape(1, ishape.ndim());
    output_dim_ = oshape.ProdShape(1, oshape.ndim());

    num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
    num_kernels_col2im_ = input_dim_;

    //LOG(INFO) << "conv_out_channels_ " << conv_out_channels_;
    //LOG(INFO) << "group_ " << group_;
    //LOG(INFO) << "conv_in_channels_ " << conv_in_channels_;
    //LOG(INFO) << "kernel_size " << param_.kernel.Size();
    //LOG(INFO) << "kernel_dim_ " << kernel_dim_;
    //LOG(INFO) << "weight_offset_ " << weight_offset_;
    //LOG(INFO) << "conv_out_spatial_dim_ " << conv_out_spatial_dim_;
    //LOG(INFO) << "col_offset_ " << col_offset_;
    //LOG(INFO) << "output_offset_ " << output_offset_;
    //LOG(INFO) << "col_buffer_size_" << col_buffer_size_;
    //LOG(INFO) << "imput_dim_ " << input_dim_;
    //LOG(INFO) << "output_dim_ " << output_dim_;
    //LOG(INFO) << "num_kernels_im2col_ " << num_kernels_im2col_;
    //LOG(INFO) << "num_kernels_col2im_ " << num_kernels_col2im_;
  }

 private:
  c3x3_16Param param_;
  index_t channel_axis_;  // channel axis of the input
  index_t channels_;  // number of channels of input image
  index_t num_spatial_axes_;  // number of spatial axes
  index_t num_;  // batch size
  index_t group_;  // number of groups
  index_t conv_out_channels_;  // number of output channels (num_filter)
  index_t conv_out_spatial_dim_;  // number of pixels of output images per channel
  index_t conv_in_channels_;  // number of input channels
  index_t kernel_dim_;  // number of input channels per group * kernel size
  index_t weight_offset_;  // number of output channels per group * kernel_dim_
  index_t col_offset_;
  index_t output_offset_;
  index_t col_buffer_size_;
  index_t input_dim_;
  index_t output_dim_;
  index_t num_kernels_im2col_;
  index_t num_kernels_col2im_;
  bool bias_term_;  // has bias term?
  bool is_1x1_;
};  // class c3x3_16Op

template<typename xpu>
Operator* CreateOp(c3x3_16Param param, int dtype,
                   std::vector<TShape> *in_shape,
                   std::vector<TShape> *out_shape,
                   Context ctx);

#if DMLC_USE_CXX11
class c3x3_16Prop : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    if (!param_.no_bias) {
      return {"data", "weight", "bias", "extra", "extra_bais"};
    } else {
      return {"data", "weight", "extra"};
    }
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    using namespace mshadow;
    param_.Init(kwargs);
    if (param_.kernel.ndim() == 1) {
      param_.layout = param_.layout? param_.layout.value() : mshadow::kNCW;
      if (param_.stride.ndim() == 0) param_.stride = Shape1(1);
      if (param_.dilate.ndim() == 0) param_.dilate = Shape1(1);
      if (param_.pad.ndim() == 0) param_.pad = Shape1(0);
    } else if (param_.kernel.ndim() == 2) {
      param_.layout = param_.layout ? param_.layout.value() : mshadow::kNCHW;
      if (param_.stride.ndim() == 0) param_.stride = Shape2(1, 1);
      if (param_.dilate.ndim() == 0) param_.dilate = Shape2(1, 1);
      if (param_.pad.ndim() == 0) param_.pad = Shape2(0, 0);
    } else {
      CHECK_EQ(param_.kernel.ndim(), 3U) << param_.kernel.ndim() << "D c3x3_16 not supported";
      param_.layout = param_.layout ? param_.layout.value(): mshadow::kNCDHW;
      if (param_.stride.ndim() == 0) param_.stride = Shape3(1, 1, 1);
      if (param_.dilate.ndim() == 0) param_.dilate = Shape3(1, 1, 1);
      if (param_.pad.ndim() == 0) param_.pad = Shape3(0, 0, 0);
    }
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    if (!param_.no_bias) {
      CHECK_EQ(in_shape->size(), 5U) << "Input:[data, weight, bias, extra, extra_weight]";
    } else {
      CHECK_EQ(in_shape->size(), 3U) << "Input:[data, weight, extra]";
    }
    // CHECK_EQ(out_shape->size(), 1) << "Output: [output]";
    out_shape->resize(1, TShape());
    const TShape &dshp = (*in_shape)[c3x3_16::kData];
    if (dshp.ndim() ==  0) return false;

    if (param_.kernel.ndim() == 2) {
      // 2d conv
      CHECK_EQ(dshp.ndim(), 4U) \
          << "Input data should be 4D in batch-num_filter-y-x";
      Shape<4> dshape = ConvertLayout(dshp.get<4>(), param_.layout.value(), kNCHW);
      Shape<4> wshape = Shape4(param_.num_filter / param_.num_group, dshape[1] / param_.num_group,
                               param_.kernel[0], param_.kernel[1]);
      wshape = ConvertLayout(wshape, kNCHW, param_.layout.value());
      wshape[0] *= param_.num_group;
      SHAPE_ASSIGN_CHECK(*in_shape, c3x3_16::kWeight, wshape);

      Shape<4> _aux_shape = Shape4((param_.num_filter * 15)/ param_.num_group, dshape[1] / param_.num_group,
                                  param_.kernel[0], param_.kernel[1]);
	  //LOG(INFO) << "------------group " << param_.num_group;
      _aux_shape[0] *= param_.num_group;
      //LOG(INFO) << "----------------" << in_shape->size();

      //LOG(INFO) << "----------------" << aux_shape->size();
      SHAPE_ASSIGN_CHECK(*in_shape, c3x3_16::kExtra, _aux_shape);
      //LOG(INFO) << "------------------------------";

      if (!param_.no_bias) {
        SHAPE_ASSIGN_CHECK(*in_shape, c3x3_16::kBias, Shape1(param_.num_filter));
        SHAPE_ASSIGN_CHECK(*in_shape, c3x3_16::kExtra_Bias, Shape1(param_.num_filter * 15));
      }

      const index_t ksize_y = static_cast<index_t>(param_.kernel[0]);
      const index_t ksize_x = static_cast<index_t>(param_.kernel[1]);
      CHECK_EQ(dshape[1] % param_.num_group, 0U) \
          << "input num_filter must divide group size";
      CHECK_EQ(param_.num_filter % param_.num_group, 0U) \
          << "output num_filter must divide group size";
      CHECK_GT(param_.kernel.Size(), 0U) \
          << "incorrect kernel size: " << param_.kernel;
      CHECK_GT(param_.stride.Size(), 0U) \
          << "incorrect stride size: " << param_.stride;
      CHECK_GT(param_.dilate.Size(), 0U) \
          << "incorrect dilate size: " << param_.dilate;

      //LOG(INFO) << "------------------------------";
      Shape<4> oshape;
      oshape[0] = dshape[0];
      oshape[1] = param_.num_filter * 15;
      oshape[2] = (dshape[2] + 2 * param_.pad[0] -
          (param_.dilate[0] * (ksize_y - 1) + 1)) / param_.stride[0] + 1;
      oshape[3] = (dshape[3] + 2 * param_.pad[1] -
          (param_.dilate[1] * (ksize_x - 1) + 1)) / param_.stride[1] + 1;
      SHAPE_ASSIGN_CHECK(*out_shape, 0, ConvertLayout(oshape, kNCHW, param_.layout.value()));
      // Perform incomplete shape inference. Fill in the missing values in data shape.
      // 1) We can always fill in the batch_size.
      // 2) We can back-calculate the input height/width if the corresponding stride is 1.
      //LOG(INFO) << "------------------------------";
      oshape = ConvertLayout((*out_shape)[0].get<4>(), param_.layout.value(), kNCHW);
      dshape[0] = oshape[0];
      if (param_.stride[0] == 1) {
        dshape[2] = oshape[2] + param_.dilate[0] * (ksize_y - 1) - 2 * param_.pad[0];
      }
      if (param_.stride[1] == 1) {
        dshape[3] = oshape[3] + param_.dilate[1] * (ksize_x - 1) - 2 * param_.pad[1];
      }
      SHAPE_ASSIGN_CHECK(*in_shape, c3x3_16::kData,
                          ConvertLayout(dshape, kNCHW, param_.layout.value()));
      // Check whether the kernel sizes are valid
      if (dshape[2] != 0) {
        CHECK_LE(ksize_y, dshape[2] + 2 * param_.pad[0]) << "kernel size exceed input";
      }
      if (dshape[3] != 0) {
        CHECK_LE(ksize_x, dshape[3] + 2 * param_.pad[1]) << "kernel size exceed input";
      }
      //LOG(INFO) << "---------------------return infer";
      return true;
    } else {
      //LOG(FATAL) << "Unknown c3x3_16 type";
      return false;
    }
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (index_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype;
      } else {
        CHECK_EQ((*in_type)[i], dtype) << "This layer requires uniform type. "
                                       << "Expected " << dtype << " v.s. given "
                                       << (*in_type)[i] << " at " << ListArguments()[i];
      }
    }
    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new c3x3_16Prop();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "c3x3_16";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[c3x3_16::kOut], in_data[c3x3_16::kData], in_data[c3x3_16::kWeight], in_data[c3x3_16::kExtra]};
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  c3x3_16Param param_;
};  // class c3x3_16Prop
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_c3x3_16_INL_H_
