// code by jph
package ch.alpine.subare.td;

import ch.alpine.subare.api.StepRecord;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;

public class PrioritizedStateAction implements Comparable<PrioritizedStateAction> {
  private final Scalar P;
  private final StepRecord stepRecord;

  public PrioritizedStateAction(Scalar P, StepRecord stepRecord) {
    this.P = P;
    this.stepRecord = stepRecord;
  }

  @Override
  public int compareTo(PrioritizedStateAction prioritizedStateAction) {
    return Scalars.compare(prioritizedStateAction.P, P);
  }

  public Tensor state() {
    return stepRecord.prevState();
  }

  public Tensor action() {
    return stepRecord.action();
  }
}
