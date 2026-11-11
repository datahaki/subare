// package ch.alpine.subare.net;
//
// import ch.alpine.tensor.RealScalar;
// import ch.alpine.tensor.Tensor;
// import ch.alpine.tensor.alg.VectorQ;
// import ch.alpine.tensor.ext.ArgMax;
//
// public class ArgMaxLayer implements NetLayer {
// Tensor inputCache;
//
// @Override
// public Tensor forward(Tensor x) {
// inputCache = x;
// return VectorQ.of(x) //
// ? RealScalar.of(ArgMax.of(x))
// : Tensor.of(inputCache.stream().map(r -> RealScalar.of(ArgMax.of(r))));
// }
//
// @Override
// public Tensor back(Tensor gradOutput) {
// return gradOutput;
// }
//
// @Override
// public NetError error(Tensor target) {
// return CrossEntropyLoss.of(inputCache, target);
// }
// }
