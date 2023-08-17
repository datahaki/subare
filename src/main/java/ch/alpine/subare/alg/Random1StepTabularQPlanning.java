// code by jph
package ch.alpine.subare.alg;

import ch.alpine.subare.api.DiscreteModel;
import ch.alpine.subare.api.QsaInterface;
import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.api.StateActionCounterSupplier;
import ch.alpine.subare.api.StepDigest;
import ch.alpine.subare.api.StepRecord;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.LearningRate;
import ch.alpine.subare.util.StateAction;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.red.Max;

/** similar to QLearning except does not use episodes but single steps
 * 
 * similar to {@link ActionValueIteration} but with gauss-seidel updates
 * therefore not parallel()
 * 
 * see box on p.169
 * 
 * algorithm performs poorly when rewards are unevenly distributed */
public class Random1StepTabularQPlanning implements StepDigest, StateActionCounterSupplier {
  public static Random1StepTabularQPlanning of(DiscreteModel discreteModel, QsaInterface qsa, LearningRate learningRate) {
    return new Random1StepTabularQPlanning(discreteModel, qsa, learningRate, new DiscreteStateActionCounter());
  }

  // ---
  private final DiscreteModel discreteModel;
  private final DiscreteQsa qsa;
  private final Scalar gamma;
  private final LearningRate learningRate;
  private final StateActionCounter sac;

  /** @param discreteModel
   * @param qsa
   * @param learningRate for deterministic tasks, a learning rate of constant == 1 is feasible */
  private Random1StepTabularQPlanning( //
      DiscreteModel discreteModel, QsaInterface qsa, LearningRate learningRate, StateActionCounter sac) {
    this.discreteModel = discreteModel;
    this.qsa = (DiscreteQsa) qsa;
    this.gamma = discreteModel.gamma();
    this.learningRate = learningRate;
    this.sac = sac;
  }

  @Override
  public void digest(StepRecord stepInterface) {
    Tensor state0 = stepInterface.prevState();
    Tensor action = stepInterface.action();
    Scalar reward = stepInterface.reward();
    Tensor state1 = stepInterface.nextState();
    // ---
    Scalar max = discreteModel.actions(state1).stream() //
        // ignore un-encountered state-action pairs, otherwise influenced by initial qsa value
        .filter(action1 -> sac.isEncountered(StateAction.key(state1, action1))) //
        .map(action1 -> qsa.value(state1, action1)) //
        .reduce(Max::of) //
        .orElse(RealScalar.ZERO);
    Scalar alpha = learningRate.alpha(stepInterface, sac);
    Scalar value1 = reward.add(gamma.multiply(max));
    qsa.blend(state0, action, value1, alpha);
    sac.digest(stepInterface);
  }

  @Override
  public StateActionCounter sac() {
    return sac;
  }
}
