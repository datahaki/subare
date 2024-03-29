// code by jph
package ch.alpine.subare.td;

import java.util.Deque;

import ch.alpine.subare.api.DiscountFunction;
import ch.alpine.subare.api.LearningRate;
import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.api.StepRecord;
import ch.alpine.subare.api.VsInterface;
import ch.alpine.subare.util.DequeDigestAdapter;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

/** n-step temporal difference for estimating V(s)
 * 
 * box on p.154 */
// TODO SUBARE not tested yet
public class NStepTemporalDifference extends DequeDigestAdapter {
  private final VsInterface vs;
  private final DiscountFunction discountFunction;
  private final LearningRate learningRate;
  private final StateActionCounter sac;

  public NStepTemporalDifference(VsInterface vs, Scalar gamma, LearningRate learningRate, StateActionCounter sac) {
    this.vs = vs;
    discountFunction = DiscountFunction.of(gamma);
    this.learningRate = learningRate;
    this.sac = sac;
  }

  @Override
  public void digest(Deque<StepRecord> deque) {
    StepRecord last = deque.getLast();
    Tensor rewards = Tensor.of(deque.stream().map(StepRecord::reward));
    rewards.append(vs.value(last.nextState()));
    // ---
    final StepRecord stepRecord = deque.getFirst(); // first step in queue
    // ---
    Tensor state0 = stepRecord.prevState();
    Scalar value0 = vs.value(state0);
    Scalar alpha = learningRate.alpha(stepRecord, sac);
    Scalar delta = discountFunction.apply(rewards).subtract(value0).multiply(alpha);
    vs.increment(state0, delta);
    sac.digest(stepRecord);
  }
}
