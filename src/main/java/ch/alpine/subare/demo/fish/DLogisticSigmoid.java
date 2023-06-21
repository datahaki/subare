// code by jph
package ch.alpine.subare.demo.fish;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.api.ScalarUnaryOperator;
import ch.alpine.tensor.sca.exp.Exp;

/* package */ enum DLogisticSigmoid implements ScalarUnaryOperator {
  FUNCTION;

  @Override
  public Scalar apply(Scalar scalar) {
    Scalar exp = Exp.FUNCTION.apply(scalar); // Exp[x]
    Scalar den = RealScalar.ONE.add(exp); // 1+Exp[x]
    return exp.divide(den.multiply(den)); // Exp[x] / (1+Exp[x])^2
  }
}
