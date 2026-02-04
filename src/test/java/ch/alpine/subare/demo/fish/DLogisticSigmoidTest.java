package ch.alpine.subare.demo.fish;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.jet.JetScalar;
import ch.alpine.tensor.mat.Tolerance;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;
import ch.alpine.tensor.sca.exp.LogisticSigmoid;

class DLogisticSigmoidTest {
  @Test
  void test() {
    Scalar t = RandomVariate.of(NormalDistribution.standard());
    JetScalar jetScalar = JetScalar.of(t, 2);
    Scalar f0 = LogisticSigmoid.FUNCTION.apply(jetScalar);
    Scalar d0 = DLogisticSigmoid.FUNCTION.apply(t);
    JetScalar js = (JetScalar) f0;
    Tolerance.CHOP.requireClose(js.vector().Get(1), d0);
  }
}
