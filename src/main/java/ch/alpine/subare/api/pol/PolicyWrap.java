// code by jph
package ch.alpine.subare.api.pol;

import ch.alpine.subare.api.mod.DiscreteModel;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.mat.Tolerance;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.d.CategoricalDistribution;
import ch.alpine.tensor.red.Total;

/** class picks action based on distribution defined by given {@link Policy} */
public record PolicyWrap(Policy policy) {
  /** @param state
   * @param actions non-empty subset of all possible actions from given state
   * @return */
  public Tensor next(Tensor state, Tensor actions) {
    Tensor pdf = Tensor.of(actions.stream().map(action -> policy.probability(state, action)));
    Tolerance.CHOP.requireClose(Total.ofVector(pdf), RealScalar.ONE);
    Distribution distribution = CategoricalDistribution.fromUnscaledPDF(pdf);
    return actions.get(Scalars.intValueExact(RandomVariate.of(distribution)));
  }

  /** @param state
   * @param discreteModel
   * @return */
  public Tensor next(Tensor state, DiscreteModel discreteModel) {
    Distribution distribution = policy.getDistribution(state);
    return discreteModel.actions(state).get(Scalars.intValueExact(RandomVariate.of(distribution)));
  }
}
