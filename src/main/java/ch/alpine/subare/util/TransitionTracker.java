// code by jph
package ch.alpine.subare.util;

import java.util.LinkedHashMap;
import java.util.Map;

import ch.alpine.subare.api.StepRecord;
import ch.alpine.subare.math.AverageTracker;
import ch.alpine.subare.pol.StepDigest;
import ch.alpine.tensor.Rational;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

/** tracks rewards and statistics of next states for a fixed (state, action) pair
 * 
 * utility class for {@link ActionValueStatistics} */
class TransitionTracker implements StepDigest {
  private final AverageTracker average = new AverageTracker();
  private final Map<Tensor, Integer> map = new LinkedHashMap<>();
  private long total = 0;

  @Override
  public void digest(StepRecord stepRecord) {
    // it is imperative that state0 and action do not change per transition tracker, maybe check this!?
    // Tensor state0 = stepRecord.prevState();
    // Tensor action = stepRecord.action();
    Scalar reward = stepRecord.reward();
    Tensor next = stepRecord.nextState();
    // ---
    average.digest(reward);
    map.merge(next, 1, Math::addExact);
    ++total;
  }

  public Scalar expectedReward() {
    return average.Get();
  }

  public Tensor transitions() {
    return Tensor.of(map.keySet().stream());
  }

  public Scalar transitionProbability(Tensor next) {
    return Rational.of(map.getOrDefault(next, 0), total);
  }
}
