// code by jph
package ch.alpine.subare.math;

import ch.alpine.tensor.DoubleScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.api.TensorUnaryOperator;
import ch.alpine.tensor.sca.Clip;
import ch.alpine.tensor.sca.tri.Cos;

/** 9.5.2 Fourier Basis p.171
 * 
 * univariate basis functions on the unit interval
 * 
 * @param order number of basis functions
 * @param clip */
public record CosineBasis(int order, Clip clip) implements TensorUnaryOperator {
  @Override // from UnaryOperator
  public Tensor apply(Tensor tensor) {
    Scalar param = clip.requireInside((Scalar) tensor);
    Scalar value = clip.rescale(param);
    return Tensors.vector(i -> Cos.FUNCTION.apply(DoubleScalar.of(i * Math.PI).multiply(value)), order);
  }
}
