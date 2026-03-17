// code by jph
package ch.alpine.subare.util;

import java.util.Set;
import java.util.stream.Collectors;

import ch.alpine.subare.api.mod.DiscreteModel;
import ch.alpine.subare.api.pol.Policy;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.num.Boole;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.RandomChoice;
import ch.alpine.tensor.pdf.d.CategoricalDistribution;

public class FixedRandomPolicy implements Policy {
  private final DiscreteModel discreteModel;
  private final Set<Tensor> set;

  public FixedRandomPolicy(DiscreteModel discreteModel) {
    this.discreteModel = discreteModel;
    set = discreteModel.states().stream() //
        .map(state -> StateAction.key(state, RandomChoice.of(discreteModel.actions(state)))) //
        .collect(Collectors.toSet());
  }

  @Override // from Policy
  public final Scalar probability(Tensor state, Tensor action) {
    return Boole.of(set.contains(StateAction.key(state, action)));
  }

  @Override // from Policy
  public Distribution getDistribution(Tensor state) {
    Tensor pdf = Tensor.of(discreteModel.actions(state).stream() //
        .map(action -> probability(state, action)));
    return CategoricalDistribution.fromUnscaledPDF(pdf);
  }
}
