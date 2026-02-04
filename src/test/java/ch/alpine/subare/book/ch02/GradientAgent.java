// code by jph
package ch.alpine.subare.book.ch02;

import java.util.Objects;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Array;
import ch.alpine.tensor.num.SoftmaxLayer;
import ch.alpine.tensor.sca.Chop;

public class GradientAgent extends Agent {
  final int n;
  final Scalar alpha;
  private final Tensor Ht;

  public GradientAgent(int n, Scalar alpha) {
    this.n = n;
    this.alpha = alpha;
    Ht = Array.zeros(n); // initially all values equal, p.38
  }

  @Override
  public int protected_takeAction() {
    Tensor pi = SoftmaxLayer.of(Ht);
    final double rnd = RANDOM.nextDouble(); // value in [0, 1)
    notifyAboutRandomizedDecision();
    double sum = 0;
    Integer a = null;
    for (int k = 0; k < n; ++k) {
      sum += pi.Get(k).number().doubleValue();
      if (rnd < sum && Objects.isNull(a))
        a = k;
    }
    Chop._10.requireClose(RealScalar.of(sum), RealScalar.ONE);
    return a;
  }

  @Override
  protected void protected_feedback(final int a, Scalar r) {
    Tensor pi = SoftmaxLayer.of(Ht);
    for (int k = 0; k < n; ++k) {
      Scalar delta = r.subtract(getRewardAverage());
      // (2.10)
      Scalar pa = pi.Get(a);
      Scalar prob = k == a //
          ? RealScalar.of(1).subtract(pa) // 1 - pi(At)
          : pa.negate(); // - pi(At)
      Ht.set(alpha.multiply(delta).multiply(prob)::add, k);
    }
  }

  @Override
  protected Tensor protected_QValues() {
    return Ht;
  }

  @Override
  public String getDescription() {
    return "a=" + alpha;
  }
}
