// code by jph
package ch.alpine.subare.mc;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ch.alpine.subare.api.DiscreteModel;
import ch.alpine.subare.api.EpisodeInterface;
import ch.alpine.subare.api.EpisodeQsaEstimator;
import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.api.StateActionCounterSupplier;
import ch.alpine.subare.api.StepRecord;
import ch.alpine.subare.math.AverageTracker;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.StateAction;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Accumulate;
import ch.alpine.tensor.alg.Last;
import ch.alpine.tensor.sca.ply.Polynomial;

/** Monte Carlo exploring starts improves an initial policy
 * based on average returns from complete episodes.
 * 
 * see box on p.107 */
public class MonteCarloExploringStarts implements EpisodeQsaEstimator, StateActionCounterSupplier {
  private final Scalar gamma;
  private final DiscreteQsa qsa;
  private final StateActionCounter sac;
  private final Map<Tensor, AverageTracker> map = new HashMap<>();

  /** @param discreteModel */
  public MonteCarloExploringStarts(DiscreteModel discreteModel) {
    this.gamma = discreteModel.gamma();
    this.qsa = DiscreteQsa.build(discreteModel); // <- "arbitrary"
    this.sac = new DiscreteStateActionCounter();
  }

  @Override // from EpisodeDigest
  public void digest(EpisodeInterface episodeInterface) {
    Map<Tensor, Integer> first = new HashMap<>();
    Tensor rewards = Tensors.empty();
    List<StepRecord> trajectory = new ArrayList<>();
    while (episodeInterface.hasNext()) {
      StepRecord stepInterface = episodeInterface.step();
      Tensor key = StateAction.key(stepInterface);
      first.computeIfAbsent(key, i -> trajectory.size());
      rewards.append(stepInterface.reward());
      trajectory.add(stepInterface);
      sac.digest(stepInterface);
    }
    Map<Tensor, Scalar> gains = new HashMap<>();
    if (gamma.equals(RealScalar.ONE)) {
      Tensor accumulate = Accumulate.of(rewards);
      for (Entry<Tensor, Integer> entry : first.entrySet()) {
        Tensor key = entry.getKey();
        Scalar alt = Last.of(accumulate);
        final int fromIndex = entry.getValue();
        if (0 < fromIndex)
          alt = alt.subtract(accumulate.Get(fromIndex - 1));
        // Scalar gain = Series.of(rewards.extract(fromIndex, rewards.length())).apply(gamma);
        // if (!gain.equals(alt))
        // throw TensorRuntimeException.of(gain, alt);
        gains.put(key, alt);
      }
    } else {
      for (Entry<Tensor, Integer> entry : first.entrySet()) {
        Tensor key = entry.getKey();
        final int fromIndex = entry.getValue();
        Scalar gain = Polynomial.of(rewards.extract(fromIndex, rewards.length())).apply(gamma);
        gains.put(key, gain);
      }
    }
    // TODO SUBARE more efficient update of average
    // compute average(Returns(s, a))
    for (StepRecord stepInterface : trajectory) {
      Tensor key = StateAction.key(stepInterface);
      // if (!map.containsKey(key))
      // map.put(key, new AverageTracker());
      map.computeIfAbsent(key, tensor -> new AverageTracker()).track(gains.get(key));
    }
    { // update
      for (Entry<Tensor, AverageTracker> entry : map.entrySet()) {
        Tensor key = entry.getKey();
        Tensor state = key.get(0);
        Tensor action = key.get(1);
        Scalar value = entry.getValue().Get();
        // System.out.println(value);
        qsa.assign(state, action, value);
      }
    }
  }

  @Override
  public DiscreteQsa qsa() {
    return qsa;
  }

  @Override
  public StateActionCounter sac() {
    return sac;
  }
}
