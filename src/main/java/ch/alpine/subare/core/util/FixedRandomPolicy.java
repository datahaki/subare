// code by jph
package ch.alpine.subare.core.util;

import java.util.Set;
import java.util.stream.Collectors;

import ch.alpine.subare.core.api.Policy;
import ch.alpine.subare.core.api.StateActionModel;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.ext.RandomChoice;
import ch.alpine.tensor.num.Boole;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.d.CategoricalDistribution;

public class FixedRandomPolicy implements Policy {
  private final StateActionModel stateActionModel;
  private final Set<Tensor> set;

  public FixedRandomPolicy(StateActionModel stateActionModel) {
    this.stateActionModel = stateActionModel;
    set = stateActionModel.states().stream() //
        .map(state -> StateAction.key(state, RandomChoice.of(stateActionModel.actions(state)))) //
        .collect(Collectors.toSet());
  }

  @Override // from Policy
  public final Scalar probability(Tensor state, Tensor action) {
    return Boole.of(set.contains(StateAction.key(state, action)));
  }

  @Override // from Policy
  public Distribution getDistribution(Tensor state) {
    Tensor pdf = Tensor.of(stateActionModel.actions(state).stream() //
        .map(action -> probability(state, action)));
    return CategoricalDistribution.fromUnscaledPDF(pdf);
  }
}
