// code by jph
package ch.alpine.subare.core.td;

import ch.alpine.subare.core.api.StepRecord;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;

public class PrioritizedStateAction implements Comparable<PrioritizedStateAction> {
  private final Scalar P;
  private final StepRecord stepInterface;

  public PrioritizedStateAction(Scalar P, StepRecord stepInterface) {
    this.P = P;
    this.stepInterface = stepInterface;
  }

  @Override
  public int compareTo(PrioritizedStateAction prioritizedStateAction) {
    return Scalars.compare(prioritizedStateAction.P, P);
  }

  public Tensor state() {
    return stepInterface.prevState();
  }

  public Tensor action() {
    return stepInterface.action();
  }
}
