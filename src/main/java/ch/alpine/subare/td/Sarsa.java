// code by jph
package ch.alpine.subare.td;

import java.util.Deque;

import ch.alpine.subare.api.DequeDigest;
import ch.alpine.subare.api.DiscountFunction;
import ch.alpine.subare.api.DiscreteModel;
import ch.alpine.subare.api.DiscreteQsaSupplier;
import ch.alpine.subare.api.LearningRate;
import ch.alpine.subare.api.PolicyExt;
import ch.alpine.subare.api.QsaInterface;
import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.api.StateActionCounterSupplier;
import ch.alpine.subare.api.StepDigest;
import ch.alpine.subare.api.StepRecord;
import ch.alpine.subare.util.DequeDigestAdapter;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.sca.Abs;

/** base class for implementations of
 * 
 * {@link OriginalSarsa}
 * {@link ExpectedSarsa}
 * {@link QLearning}
 * 
 * https://github.com/idsc-frazzoli/subare/files/2257711/sarsa.pdf
 * 
 * the abstract class Sarsa provides the capability to digest:
 * 
 * a single step {@link StepDigest},
 * as well as N-steps {@link DequeDigest} */
public class Sarsa extends DequeDigestAdapter implements DiscreteQsaSupplier, StateActionCounterSupplier {
  private final SarsaEvaluation sarsaEvaluation;
  private final DiscountFunction discountFunction;
  private final QsaInterface qsa;
  private final StateActionCounter sac;
  private final LearningRate learningRate;
  private final PolicyExt policy;

  /** @param sarsaEvaluation
   * @param discreteModel
   * @param learningRate
   * @param qsa
   * @param sac
   * @param policy */
  protected Sarsa( //
      SarsaEvaluation sarsaEvaluation, //
      DiscreteModel discreteModel, //
      LearningRate learningRate, //
      QsaInterface qsa, //
      StateActionCounter sac, //
      PolicyExt policy) {
    this.sarsaEvaluation = sarsaEvaluation;
    discountFunction = DiscountFunction.of(discreteModel.gamma());
    this.sac = sac;
    this.qsa = qsa;
    this.learningRate = learningRate;
    this.policy = policy;
  }

  @Override // from DequeDigest
  public final void digest(Deque<StepRecord> deque) {
    Tensor rewards = Tensor.of(deque.stream().map(StepRecord::reward));
    Tensor nextState = deque.getLast().nextState();
    // ---
    // for terminal state in queue, "=last.next"
    // ---
    final StepRecord stepRecord = deque.getFirst(); // first step in queue
    Tensor state0 = stepRecord.prevState();
    Tensor action = stepRecord.action();
    Scalar alpha = learningRate.alpha(stepRecord, sac);
    rewards.append(sarsaEvaluation.evaluate(nextState, policy)); // <- evaluate(...) is called here
    Scalar value1 = discountFunction.apply(rewards);
    qsa.blend(state0, action, value1, alpha);
    sac.digest(stepRecord);
  }

  /** @param stepRecord
   * @return non-negative priority rating */
  final Scalar priority(StepRecord stepRecord) {
    Tensor state0 = stepRecord.prevState();
    Tensor action = stepRecord.action();
    Scalar value0 = qsa.value(state0, action);
    Tensor rewards = Tensors.of(stepRecord.reward(), sarsaEvaluation.evaluate(stepRecord.nextState(), policy));
    return Abs.between(discountFunction.apply(rewards), value0);
  }

  @Override // from DiscreteQsaSupplier
  public final DiscreteQsa qsa() {
    return (DiscreteQsa) qsa;
  }

  @Override
  public StateActionCounter sac() {
    return sac;
  }
}
