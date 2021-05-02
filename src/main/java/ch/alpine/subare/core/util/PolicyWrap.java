// code by jph
package ch.alpine.subare.core.util;

import ch.alpine.subare.core.Policy;
import ch.alpine.subare.core.StateActionModel;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.mat.Tolerance;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.EmpiricalDistribution;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.red.Total;

/** class picks action based on distribution defined by given {@link Policy} */
public class PolicyWrap {
  private final Policy policy;

  public PolicyWrap(Policy policy) {
    this.policy = policy;
  }

  /** @param state
   * @param actions non-empty subset of all possible actions from given state
   * @return */
  public Tensor next(Tensor state, Tensor actions) {
    Tensor pdf = Tensor.of(actions.stream().map(action -> policy.probability(state, action)));
    Tolerance.CHOP.requireClose(Total.ofVector(pdf), RealScalar.ONE);
    Distribution distribution = EmpiricalDistribution.fromUnscaledPDF(pdf);
    return actions.get(Scalars.intValueExact(RandomVariate.of(distribution)));
  }

  /** @param state
   * @param stateActionModel
   * @return */
  public Tensor next(Tensor state, StateActionModel stateActionModel) {
    Distribution distribution = policy.getDistribution(state);
    return stateActionModel.actions(state).get(Scalars.intValueExact(RandomVariate.of(distribution)));
  }
}
