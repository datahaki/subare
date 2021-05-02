// code by jph
package ch.alpine.subare.core.mc;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ch.alpine.subare.core.DiscountFunction;
import ch.alpine.subare.core.DiscreteModel;
import ch.alpine.subare.core.EpisodeInterface;
import ch.alpine.subare.core.EpisodeVsEstimator;
import ch.alpine.subare.core.StepInterface;
import ch.alpine.subare.core.util.DiscreteVs;
import ch.alpine.subare.util.AverageTracker;
import ch.alpine.subare.util.Index;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Array;

/** estimates the state value function for a given policy
 * see box on p.100
 * 
 * the policy is not visible to the method.
 * 
 * the policy is only used to generate episodes that are then digested by the method.
 * 
 * for the gambler problem. */
public class FirstVisitPolicyEvaluation implements EpisodeVsEstimator {
  private final DiscreteModel discreteModel;
  // private final DiscreteVs vs;
  private final Map<Tensor, AverageTracker> map = new HashMap<>(); // TODO no good!
  private final DiscountFunction discountFunction;

  public FirstVisitPolicyEvaluation(DiscreteModel discreteModel, DiscreteVs vs) {
    this.discreteModel = discreteModel;
    // this.vs = vs; // TODO write results directly in vs!
    discountFunction = DiscountFunction.of(discreteModel.gamma());
  }

  @Override
  public void digest(EpisodeInterface episodeInterface) {
    Map<Tensor, Integer> first = new HashMap<>();
    Map<Tensor, Scalar> gains = new HashMap<>();
    Tensor rewards = Tensors.empty();
    List<StepInterface> trajectory = new ArrayList<>();
    while (episodeInterface.hasNext()) {
      StepInterface stepInterface = episodeInterface.step();
      Tensor state = stepInterface.prevState();
      if (!first.containsKey(state))
        first.put(state, trajectory.size());
      rewards.append(stepInterface.reward());
      trajectory.add(stepInterface);
    }
    for (Entry<Tensor, Integer> entry : first.entrySet()) {
      Tensor state = entry.getKey();
      int fromIndex = entry.getValue();
      gains.put(state, discountFunction.apply(rewards.extract(fromIndex, rewards.length())));
    }
    // TODO more efficient update of average
    for (StepInterface stepInterface : trajectory) {
      Tensor stateP = stepInterface.prevState();
      if (!map.containsKey(stateP))
        map.put(stateP, new AverageTracker());
      map.get(stateP).track(gains.get(stateP));
    }
  }

  @Override
  public DiscreteVs vs() {
    Tensor states = discreteModel.states();
    Index index = Index.build(states);
    Tensor values = Array.zeros(index.size());
    for (Entry<Tensor, AverageTracker> entry : map.entrySet())
      values.set(entry.getValue().getScalar(), index.of(entry.getKey()));
    return new DiscreteVs(index, values);
  }
}
