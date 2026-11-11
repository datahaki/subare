//// code by jph
// package ch.alpine.subare.net;
//
// import ch.alpine.tensor.Tensor;
// import ch.alpine.tensor.alg.VectorQ;
// import ch.alpine.tensor.nrm.SoftmaxVector;
//
/// ** inspired by
// * <a href="https://reference.wolfram.com/language/ref/SoftmaxLayer.html">SoftmaxLayer</a> */
// public class SoftmaxLayer implements NetLayer {
// // Tensor outputCache;
// @Override
// public Tensor forward(Tensor x) {
// // return outputCache = SoftmaxVector.of(x);
// return VectorQ.of(x) //
// ? SoftmaxVector.of(x)
// : Tensor.of(x.stream().map(SoftmaxVector::of));
// }
//
// @Override
// public Tensor back(Tensor gradOutput) {
// // Scalar dot = (Scalar) gradOutput.dot(outputCache);
// // return Entrywise.mul().apply(outputCache, gradOutput.maps(s -> s.subtract(dot)));
// return gradOutput;
// }
// }
