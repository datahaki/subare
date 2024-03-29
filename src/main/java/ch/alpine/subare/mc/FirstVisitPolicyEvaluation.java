// code by jph
package ch.alpine.subare.mc;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ch.alpine.subare.api.DiscountFunction;
import ch.alpine.subare.api.DiscreteModel;
import ch.alpine.subare.api.EpisodeInterface;
import ch.alpine.subare.api.EpisodeVsEstimator;
import ch.alpine.subare.api.StepRecord;
import ch.alpine.subare.math.AverageTracker;
import ch.alpine.subare.math.Index;
import ch.alpine.subare.util.DiscreteVs;
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
  private final Map<Tensor, AverageTracker> map = new HashMap<>(); // TODO SUBARE no good!
  private final DiscountFunction discountFunction;

  public FirstVisitPolicyEvaluation(DiscreteModel discreteModel, DiscreteVs vs) {
    this.discreteModel = discreteModel;
    // this.vs = vs; // TODO SUBARE write results directly in vs!
    discountFunction = DiscountFunction.of(discreteModel.gamma());
  }

  @Override
  public void digest(EpisodeInterface episodeInterface) {
    Map<Tensor, Integer> first = new HashMap<>();
    Map<Tensor, Scalar> gains = new HashMap<>();
    Tensor rewards = Tensors.empty();
    List<StepRecord> trajectory = new ArrayList<>();
    while (episodeInterface.hasNext()) {
      StepRecord stepRecord = episodeInterface.step();
      Tensor state = stepRecord.prevState();
      first.computeIfAbsent(state, i -> trajectory.size());
      rewards.append(stepRecord.reward());
      trajectory.add(stepRecord);
    }
    for (Entry<Tensor, Integer> entry : first.entrySet()) {
      Tensor state = entry.getKey();
      int fromIndex = entry.getValue();
      gains.put(state, discountFunction.apply(rewards.extract(fromIndex, rewards.length())));
    }
    // TODO SUBARE more efficient update of average
    for (StepRecord stepRecord : trajectory) {
      Tensor stateP = stepRecord.prevState();
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
      values.set(entry.getValue().Get(), index.of(entry.getKey()));
    return new DiscreteVs(index, values);
  }
}
