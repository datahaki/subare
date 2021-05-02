// code by jph
package ch.ethz.idsc.subare.core.adapter;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.red.KroneckerDelta;
import ch.ethz.idsc.subare.core.SampleModel;
import ch.ethz.idsc.subare.core.StandardModel;

/** applicable for models that have deterministic move and reward */
public abstract class DeterministicStandardModel implements StandardModel, SampleModel {
  @Override
  public final Scalar expectedReward(Tensor state, Tensor action) {
    // reward(s, a, s') == expectedReward(s, a)
    return reward(state, action, move(state, action)); // deterministic reward
  }

  @Override
  public final Tensor transitions(Tensor state, Tensor action) {
    return Tensors.of(move(state, action)); // deterministic transition
  }

  @Override
  public final Scalar transitionProbability(Tensor state, Tensor action, Tensor next) {
    return KroneckerDelta.of(move(state, action), next);
  }
}
