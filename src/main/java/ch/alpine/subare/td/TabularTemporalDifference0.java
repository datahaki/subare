// code by jph
package ch.alpine.subare.td;

import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.api.StepDigest;
import ch.alpine.subare.api.StepRecord;
import ch.alpine.subare.api.VsInterface;
import ch.alpine.subare.util.LearningRate;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.sca.Clips;

/** Tabular TD(0) for estimating value of policy
 * using eq (6.2) on p.95
 * <pre>
 * V(S) = V(S) + alpha * [R + gamma * V(S') - V(S)]
 * </pre>
 * 
 * Implementation also covers
 * Semi-gradient TD(0) for estimating an approximate value function
 * in 9.3, p. 164
 * 
 * @param vs
 * @param gamma discount factor
 * @param learningRate */
public record TabularTemporalDifference0( //
    VsInterface vs, //
    Scalar gamma, //
    LearningRate learningRate, //
    StateActionCounter sac) implements StepDigest {
  public TabularTemporalDifference0 {
    Clips.unit().requireInside(gamma);
  }

  @Override // from StepDigest
  public void digest(StepRecord stepInterface) {
    Tensor state0 = stepInterface.prevState();
    // action is only required for learning rate
    Scalar reward = stepInterface.reward();
    Tensor state1 = stepInterface.nextState();
    // ---
    Scalar value0 = vs.value(state0);
    Scalar value1 = vs.value(state1);
    Scalar alpha = learningRate.alpha(stepInterface, sac);
    Scalar delta = reward.add(gamma.multiply(value1)).subtract(value0).multiply(alpha);
    vs.increment(state0, delta);
    sac.digest(stepInterface);
  }
}
