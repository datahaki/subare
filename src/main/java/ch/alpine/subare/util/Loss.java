// code by jph
package ch.alpine.subare.util;

import ch.alpine.subare.api.DiscreteModel;
import ch.alpine.subare.math.FairArg;
import ch.alpine.tensor.RationalScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.nrm.Vector1Norm;
import ch.alpine.tensor.red.Max;
import ch.alpine.tensor.sca.Sign;

/** measures to compare performance between optimal state-action value function and learned q-function
 * 
 * loss is non-negative
 * positive loss means deviation from optimal policy
 * 
 * if the reference qsa is in exact precision, the return values also have exact precision
 * 
 * loss can be measured per state-action, per state, and overall accumulated as a single number */
public enum Loss {
  ;
  /** @param discreteModel
   * @param ref ground truth
   * @param qsa
   * @return */
  public static DiscreteQsa asQsa(DiscreteModel discreteModel, DiscreteQsa ref, DiscreteQsa qsa) {
    DiscreteQsa loss = DiscreteQsa.build(discreteModel);
    for (Tensor state : discreteModel.states()) {
      Tensor actions = discreteModel.actions(state);
      Scalar max = actions.stream() //
          .map(action -> ref.value(state, action)) //
          .reduce(Max::of).get();
      // ---
      FairArg fairArg = FairArg.max(Tensor.of(actions.stream().map(action -> qsa.value(state, action))));
      Scalar weight = RationalScalar.of(1, fairArg.optionsCount());
      for (int index : fairArg.options()) {
        Tensor action = actions.get(index);
        Scalar delta = max.subtract(ref.value(state, action));
        Sign.requirePositiveOrZero(delta);
        loss.assign(state, action, delta.multiply(weight));
      }
    }
    return loss;
  }

  /** @param discreteModel
   * @param ref ground truth
   * @param qsa
   * @return map that assigns each state a non-negative number which is the deficiency
   * of the evaluation of the state from the optimal evaluation */
  public static DiscreteVs perState(DiscreteModel discreteModel, DiscreteQsa ref, DiscreteQsa qsa) {
    return DiscreteUtils.reduce(discreteModel, asQsa(discreteModel, ref, qsa), Scalar::add);
  }

  /** @param discreteModel
   * @param ref ground truth
   * @param qsa
   * @return non-negative number which should be subtracted from the optimal gains */
  public static Scalar accumulation(DiscreteModel discreteModel, DiscreteQsa ref, DiscreteQsa qsa) {
    return Vector1Norm.of(perState(discreteModel, ref, qsa).values());
  }
}
