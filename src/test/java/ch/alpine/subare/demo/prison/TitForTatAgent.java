// code by jph
package ch.alpine.subare.demo.prison;

import ch.alpine.subare.ch02.Agent;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;

/* package */ class TitForTatAgent extends Agent {
  private int nextAction = 1;

  @Override
  public int protected_takeAction() {
    return nextAction;
  }

  @Override
  protected Tensor protected_QValues() {
    return Tensors.vectorInt(-1, -1);
  }

  @Override
  protected void protected_feedback(int a, Scalar value) {
    Tensor rew = Training.R2.get(a);
    if (rew.Get(0).equals(value)) {
      nextAction = 0;
      return;
    }
    if (rew.Get(1).equals(value)) {
      nextAction = 1;
      return;
    }
    throw new RuntimeException();
  }

  @Override
  public String getDescription() {
    return "";
  }
}
