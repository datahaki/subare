// code by jph
package ch.alpine.subare.ch05.infvar;

import ch.alpine.subare.core.Policy;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.pdf.Distribution;

/* package */ class ConstantPolicy implements Policy {
  final Scalar backProb;

  public ConstantPolicy(Scalar backProb) {
    this.backProb = backProb;
  }

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
