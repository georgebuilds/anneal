package onnx

// Handler registry. Device-tier handlers populated for Phase 1.B (CNN core).
// Host-tier handlers register from their own init blocks; this file only
// wires the device dispatch.
//
// Coverage: Stage-1 CNN core per plan §5: Conv, BatchNormalization, Relu,
// Clip, Add, MaxPool, GlobalAveragePool, Gemm, Flatten, Reshape, Constant,
// plus the classifier-tail glue (Shape, Gather, Concat, Unsqueeze) and the
// Stage-2 down-payment (Sub/Mul/Div/Pow/Sqrt/Neg/Tanh/Sigmoid/Cast/Equal/
// MatMul/Transpose/Squeeze/Slice/Expand/ReduceSum/Mean/Max).

// RegisterAll installs every canonical device-tier handler on r.
func RegisterAll(r *Runner) {
	// const + identity
	r.RegisterHandler("Constant", handleConstant)
	r.RegisterHandler("Identity", handleIdentity)
	r.RegisterHandler("ConstantOfShape", handleConstantOfShape)

	// elementwise
	r.RegisterHandler("Add", handleAdd)
	r.RegisterHandler("Sub", handleSub)
	r.RegisterHandler("Mul", handleMul)
	r.RegisterHandler("Div", handleDiv)
	r.RegisterHandler("Pow", handlePow)
	r.RegisterHandler("Sqrt", handleSqrt)
	r.RegisterHandler("Neg", handleNeg)
	r.RegisterHandler("Tanh", handleTanh)
	r.RegisterHandler("Sigmoid", handleSigmoid)
	r.RegisterHandler("Relu", handleRelu)
	r.RegisterHandler("Clip", handleClip)
	r.RegisterHandler("Cast", handleCast)
	r.RegisterHandler("Equal", handleEqual)

	// reduction
	r.RegisterHandler("ReduceSum", handleReduceSum)
	r.RegisterHandler("ReduceMean", handleReduceMean)
	r.RegisterHandler("ReduceMax", handleReduceMax)

	// movement
	r.RegisterHandler("Reshape", handleReshape)
	r.RegisterHandler("Flatten", handleFlatten)
	r.RegisterHandler("Squeeze", handleSqueeze)
	r.RegisterHandler("Unsqueeze", handleUnsqueeze)
	r.RegisterHandler("Transpose", handleTranspose)
	r.RegisterHandler("Concat", handleConcat)
	r.RegisterHandler("Gather", handleGather)
	r.RegisterHandler("Slice", handleSlice)
	r.RegisterHandler("Expand", handleExpand)

	// conv / pool
	r.RegisterHandler("Conv", handleConv)
	r.RegisterHandler("MaxPool", handleMaxPool)
	r.RegisterHandler("GlobalAveragePool", handleGlobalAveragePool)

	// norm
	r.RegisterHandler("BatchNormalization", handleBatchNormalization)

	// linear
	r.RegisterHandler("Gemm", handleGemm)
	r.RegisterHandler("MatMul", handleMatMul)
}
