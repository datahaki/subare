// code by jph
package ch.ethz.idsc.subare.ch02;

import ch.alpine.tensor.RationalScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;

/** the random agent picks any action equally likely
 * the policy is a constant vector of pi(a)=1/n */
public class RandomAgent extends Agent {
  final int n;

  public RandomAgent(int n) {
    this.n = n;
  }

  @Override
  public int protected_takeAction() {
    notifyAboutRandomizedDecision();
    return RANDOM.nextInt(n);
  }

  @Override
  protected void protected_feedback(int a, Scalar value) {
    // ---
  }

  @Override
  protected Tensor protected_QValues() {
    return Tensors.vector(i -> RationalScalar.of(1, n), n);
  }

  @Override
  public String getDescription() {
    return "";
  }
}
