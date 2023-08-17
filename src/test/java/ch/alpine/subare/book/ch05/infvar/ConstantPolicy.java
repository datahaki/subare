// code by jph
package ch.alpine.subare.book.ch05.infvar;

import ch.alpine.subare.api.Policy;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.pdf.Distribution;

/* package */ record ConstantPolicy(Scalar backProb) implements Policy {
  @Override
  public Scalar probability(Tensor state, Tensor action) {
    if (state.equals(RealScalar.ZERO))
      return action.equals(RealScalar.ZERO) //
          ? backProb
          : RealScalar.ONE.subtract(backProb);
    return RealScalar.ONE;
  }

  @Override
  public Distribution getDistribution(Tensor state) {
    throw new UnsupportedOperationException();
  }
}
