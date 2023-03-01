// code by jph
package ch.alpine.subare.core.adapter;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.random.RandomGenerator;

import ch.alpine.subare.core.Policy;
import ch.alpine.subare.core.StateActionModel;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.num.Boole;
import ch.alpine.tensor.pdf.Distribution;

public class FixedRandomPolicy implements Policy {
  private static final RandomGenerator RANDOM = new Random();
  // ---
  private final Set<Tensor> set = new HashSet<>();

  public FixedRandomPolicy(StateActionModel stateActionModel) {
    for (Tensor state : stateActionModel.states()) {
      Tensor actions = stateActionModel.actions(state);
      set.add(Tensors.of(state, actions.get(RANDOM.nextInt(actions.length()))));
    }
  }

  @Override // from Policy
  public final Scalar probability(Tensor state, Tensor action) {
    return Boole.of(set.contains(Tensors.of(state, action)));
  }

  @Override
  public Distribution getDistribution(Tensor state) {
    throw new UnsupportedOperationException();
  }
}
