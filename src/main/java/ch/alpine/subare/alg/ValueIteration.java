// code by jph
package ch.alpine.subare.alg;

import java.util.Objects;

import ch.alpine.subare.api.ActionValueInterface;
import ch.alpine.subare.api.DiscreteModel;
import ch.alpine.subare.api.DiscreteVsSupplier;
import ch.alpine.subare.api.StandardModel;
import ch.alpine.subare.api.VsInterface;
import ch.alpine.subare.util.ActionValueAdapter;
import ch.alpine.subare.util.DiscreteValueFunctions;
import ch.alpine.subare.util.DiscreteVs;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.qty.Quantity;
import ch.alpine.tensor.qty.Timing;
import ch.alpine.tensor.red.Max;

/** value iteration: "policy evaluation is stopped after just one sweep"
 * eq (3.14) in 3.5, p.46
 * eq (4.10) in 4.4, p.65
 * see box in 4.4, on p.65
 * 
 * approximately equivalent to iterating with {@link GreedyPolicy}
 * 
 * parallel implementation
 * initial values are set to zeros
 * Jacobi style, i.e. updates take effect only in the next iteration */
public class ValueIteration implements DiscreteVsSupplier {
  /** @param standardModel
   * @param threshold
   * @return */
  public static DiscreteVs solve(StandardModel standardModel, Scalar threshold) {
    ValueIteration valueIteration = new ValueIteration(standardModel);
    valueIteration.untilBelow(threshold);
    return valueIteration.vs();
  }

  // ---
  private final DiscreteModel discreteModel;
  private final ActionValueAdapter actionValueAdapter;
  private final Scalar gamma;
  private DiscreteVs vs_new;
  private DiscreteVs vs_old;
  private int iterations = 0;
  private int alternate = 0;

  /** @param standardModel */
  public ValueIteration(StandardModel standardModel) {
    this(standardModel, standardModel);
  }

  /** @param standardModel */
  public ValueIteration(DiscreteModel discreteModel, ActionValueInterface actionValueInterface) {
    this.discreteModel = discreteModel;
    actionValueAdapter = new ActionValueAdapter(actionValueInterface);
    this.gamma = discreteModel.gamma();
    vs_new = DiscreteVs.build(discreteModel.states());
  }

  /** perform iteration until values don't change more than threshold
   * 
   * @param threshold
   * @return */
  public void untilBelow(Scalar threshold) {
    untilBelow(threshold, Integer.MAX_VALUE);
  }

  private static final Scalar LIMIT = Quantity.of(3e9, "ns");

  public void untilBelow(Scalar threshold, int flips) {
    Scalar past = null;
    Timing timing = Timing.started();
    while (true) {
      step();
      final Scalar delta = DiscreteValueFunctions.distance(vs_new, vs_old);
      if (Scalars.lessThan(LIMIT, timing.nanoSeconds()))
        IO.println(past + " -> " + delta + " " + alternate);
      if (Objects.nonNull(past) && Scalars.lessThan(past, delta))
        if (flips < ++alternate) {
          IO.println("give up at " + past + " -> " + delta);
          break;
        }
      past = delta;
      if (Scalars.lessThan(delta, threshold))
        break;
    }
  }

  /** perform one step of the iteration
   * 
   * @return */
  public void step() {
    vs_old = vs_new.copy();
    VsInterface discounted = vs_new.discounted(gamma);
    vs_new = vs_new.create(vs_new.keys().stream() //
        .parallel() //
        .map(state -> jacobiMax(state, discounted)));
    ++iterations;
  }

  private Scalar jacobiMax(Tensor state, VsInterface gvalues) {
    return discreteModel.actions(state).stream() //
        .map(action -> actionValueAdapter.qsa(state, action, gvalues)) //
        .reduce(Max::of) //
        .orElseThrow();
  }

  @Override
  public DiscreteVs vs() {
    return vs_new;
  }

  public int iterations() {
    return iterations;
  }
}
