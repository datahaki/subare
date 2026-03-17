// code by jph
// inspired by Shangtong Zhang
package ch.alpine.subare.alg;

import java.util.Objects;

import ch.alpine.subare.api.DiscreteVsSupplier;
import ch.alpine.subare.api.VsInterface;
import ch.alpine.subare.api.mod.StandardModel;
import ch.alpine.subare.api.pol.Policy;
import ch.alpine.subare.util.ActionValueAdapter;
import ch.alpine.subare.util.DiscreteValueFunctions;
import ch.alpine.subare.util.DiscreteVs;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.qty.Quantity;
import ch.alpine.tensor.qty.Timing;
import ch.alpine.tensor.sca.Chop;
import ch.alpine.tensor.sca.N;

/** general bellman equation:
 * v_pi(s) == Sum_a pi(a|s) * Sum_{s', r} p(s', r | s, a) * (r + gamma * v_pi(s'))
 * bellman optimality equation:
 * v_*(s) == max_a Sum_{s', r} p(s', r | s, a) * (r + gamma * v_*(s')) */
public class IterativePolicyEvaluation extends BaseIteration implements DiscreteVsSupplier {
  private final StandardModel standardModel;
  private final ActionValueAdapter actionValueAdapter;
  private final Policy policy;
  private final Scalar gamma;
  private DiscreteVs vs_new;
  private DiscreteVs vs_old;
  private int alternate = 0;

  // ---
  /** iterative policy evaluation (4.5)
   * see box on p.81
   * 
   * parallel implementation
   * initial values are set to zeros
   * Jacobi style, i.e. updates take effect only in the next iteration
   * 
   * @param standardModel
   * @param policy
   * @return */
  public IterativePolicyEvaluation(StandardModel standardModel, Policy policy) {
    this.standardModel = standardModel;
    actionValueAdapter = new ActionValueAdapter(standardModel);
    this.policy = policy;
    this.gamma = standardModel.gamma();
    vs_new = DiscreteVs.build(standardModel.states());
  }

  /** @param threshold
   * @return */
  public void until(Chop chop) {
    until(chop, Integer.MAX_VALUE);
  }

  private static final Scalar LIMIT = Quantity.of(3e9, "ns");

  public void until(Chop chop, int flips) {
    Scalar past = null;
    Timing timing = Timing.started();
    while (true) {
      step();
      Scalar delta = DiscreteValueFunctions.distance(vs_new, vs_old);
      appendRow(delta);
      if (Scalars.lessThan(LIMIT, timing.nanoSeconds()))
        IO.println(past + " -> " + delta + " " + alternate);
      if (Objects.nonNull(past) && Scalars.lessThan(past, delta))
        if (flips < ++alternate) {
          IO.println("give up at " + past + " -> " + delta);
          break;
        }
      if (chop.isZero(N.DOUBLE.apply(delta)))
        break;
      past = delta;
    }
  }

  public void step() {
    vs_old = vs_new.copy();
    VsInterface discounted = vs_new.discounted(gamma);
    vs_new = vs_new.create(standardModel.states().stream() //
        .parallel() //
        .map(state -> jacobiAdd(state, discounted)));
  }

  // helper function
  private Scalar jacobiAdd(Tensor state, VsInterface gvalues) {
    return standardModel.actions(state).stream() //
        .map(action -> policy.probability(state, action).multiply( //
            actionValueAdapter.qsa(state, action, gvalues))) //
        .reduce(Scalar::add) //
        .orElseThrow();
  }

  @Override
  public DiscreteVs vs() {
    return vs_new;
  }
}
