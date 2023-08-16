// code by jph
package ch.alpine.subare.core.td;

import java.util.Deque;

import ch.alpine.subare.core.DiscountFunction;
import ch.alpine.subare.core.DiscreteModel;
import ch.alpine.subare.core.DiscreteQsaSupplier;
import ch.alpine.subare.core.QsaInterface;
import ch.alpine.subare.core.StateActionCounter;
import ch.alpine.subare.core.StateActionCounterSupplier;
import ch.alpine.subare.core.StepRecord;
import ch.alpine.subare.core.adapter.DequeDigestAdapter;
import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.subare.core.util.DiscreteValueFunctions;
import ch.alpine.subare.core.util.LearningRate;
import ch.alpine.subare.core.util.PolicyBase;
import ch.alpine.subare.core.util.PolicyExt;
import ch.alpine.subare.core.util.StateActionCounterUtil;
import ch.alpine.subare.util.Coinflip;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

/** double sarsa for single-step, and n-step
 * 
 * implementation covers
 * 
 * Double Q-learning (box on p.145)
 * Double Expected Sarsa (Exercise 6.10 p.145)
 * Double Original Sarsa (p.145)
 * 
 * the update equation in the box uses argmax_a.
 * since there may not be a unique best action, we average the evaluation
 * over all best actions.
 * 
 * Maximization bias and Doubled learning were introduced and investigated
 * by Hado van Hasselt (2010, 2011) */
public class DoubleSarsa extends DequeDigestAdapter implements DiscreteQsaSupplier, StateActionCounterSupplier {
  private final Coinflip coinflip = Coinflip.fair();
  // ---
  private final DiscreteModel discreteModel;
  private final DiscountFunction discountFunction;
  private final SarsaEvaluation sarsaEvaluation;
  private final QsaInterface qsa1;
  private final QsaInterface qsa2;
  private final LearningRate learningRate;
  private final StateActionCounter sac1;
  private final StateActionCounter sac2;
  private final PolicyExt policy1;
  private final PolicyExt policy2;

  /** @param sarsaEvaluation
   * @param discreteModel
   * @param learningRate
   * @param qsa1
   * @param qsa2
   * @param sac1
   * @param sac2 */
  public DoubleSarsa( //
      SarsaEvaluation sarsaEvaluation, //
      DiscreteModel discreteModel, //
      LearningRate learningRate, //
      QsaInterface qsa1, //
      QsaInterface qsa2, //
      StateActionCounter sac1, //
      StateActionCounter sac2, //
      PolicyExt policy1, //
      PolicyExt policy2) {
    this.discreteModel = discreteModel;
    discountFunction = DiscountFunction.of(discreteModel.gamma());
    this.sarsaEvaluation = sarsaEvaluation;
    this.qsa1 = qsa1;
    this.qsa2 = qsa2;
    this.learningRate = learningRate;
    this.sac1 = sac1;
    this.sac2 = sac2;
    this.policy1 = policy1;
    this.policy2 = policy2;
  }

  /** @return policy with respect to (qsa1 + qsa2) / 2 and sac1+sac2 */
  public PolicyBase getPolicy() {
    PolicyBase copy = (PolicyBase) policy1.copy();
    copy.setQsa(DiscreteValueFunctions.average((DiscreteQsa) qsa1, (DiscreteQsa) qsa2));
    copy.setSac(StateActionCounterUtil.getSummedSac(sac1, sac2, discreteModel));
    return copy;
  }

  @Override // from DequeDigest
  public void digest(Deque<StepRecord> deque) {
    // randomly select which qsa to read and write
    boolean flip = coinflip.tossHead(); // flip coin, probability 0.5 each
    PolicyExt Policy1 = flip ? policy2 : policy1;
    PolicyExt Policy2 = flip ? policy1 : policy2;
    // ---
    Tensor rewards = Tensor.of(deque.stream().map(StepRecord::reward));
    Tensor nextState = deque.getLast().nextState();
    Scalar expectedReward = sarsaEvaluation.crossEvaluate(nextState, Policy1, Policy2);
    rewards.append(expectedReward);
    // ---
    // the code below is identical to Sarsa
    StepRecord first = deque.getFirst();
    Tensor state0 = first.prevState(); // state-action pair that is being updated in Q
    Tensor action0 = first.action();
    Scalar alpha = learningRate.alpha(first, Policy1.sac());
    Scalar gain = discountFunction.apply(rewards);
    Policy1.qsaInterface().blend(state0, action0, gain, alpha); // update Qsa1
    Policy1.sac().digest(first); // signal to LearningRate1
  }

  @Override // from DiscreteQsaSupplier
  public DiscreteQsa qsa() {
    return DiscreteValueFunctions.weightedAverage( //
        (DiscreteQsa) qsa1, (DiscreteQsa) qsa2, sac1, sac2);
  }

  @Override // from StateActionCounterSupplier
  public StateActionCounter sac() {
    return StateActionCounterUtil.getSummedSac(sac1, sac2, discreteModel);
  }
}
