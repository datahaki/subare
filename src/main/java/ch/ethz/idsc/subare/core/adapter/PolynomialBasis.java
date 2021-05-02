// code by jph
package ch.ethz.idsc.subare.core.adapter;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.NestList;
import ch.alpine.tensor.api.TensorUnaryOperator;
import ch.alpine.tensor.sca.Clip;

/** 9.5.2 Fourier Basis p.171
 * 
 * univariate basis functions on the unit interval */
public class PolynomialBasis implements TensorUnaryOperator {
  /** @param order number of basis functions
   * @param clip */
  public static TensorUnaryOperator create(int order, Clip clip) {
    return new PolynomialBasis(order, clip);
  }

  /***************************************************/
  private final int order;
  private final Clip clip;

  private PolynomialBasis(int order, Clip clip) {
    this.order = order;
    this.clip = clip;
  }

  @Override // from UnaryOperator
  public Tensor apply(Tensor tensor) {
    Scalar param = clip.requireInside((Scalar) tensor);
    Scalar value = clip.rescale(param);
    return NestList.of(value::multiply, RealScalar.ONE, order - 1);
  }
}
