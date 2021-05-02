// code by jph
package ch.alpine.subare.ch02;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.UnitVector;

/** an agent that always produces the same predefined action */
class ConstantAgent extends Agent {
  final int n;
  final int action;

  public ConstantAgent(int n, int action) {
    this.n = n;
    this.action = action;
  }

  @Override
  public int protected_takeAction() {
    return action;
  }

  @Override
  protected void protected_feedback(int a, Scalar value) {
    // ---
  }

  @Override
  public String getDescription() {
    return "A=" + action;
  }

  @Override
  protected Tensor protected_QValues() {
    return UnitVector.of(n, action);
  }
}
