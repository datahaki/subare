// code by jph
// inspired by Shangtong Zhang
package ch.alpine.subare.alg;

import java.util.Objects;

import ch.alpine.subare.api.Policy;
import ch.alpine.subare.api.StandardModel;
import ch.alpine.subare.api.VsInterface;
import ch.alpine.subare.util.ActionValueAdapter;
import ch.alpine.subare.util.DiscreteValueFunctions;
import ch.alpine.subare.util.DiscreteVs;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.ext.Timing;

/** general bellman equation:
 * v_pi(s) == Sum_a pi(a|s) * Sum_{s', r} p(s', r | s, a) * (r + gamma * v_pi(s'))
 * bellman optimality equation:
 * v_*(s) == max_a Sum_{s', r} p(s', r | s, a) * (r + gamma * v_*(s')) */
public class IterativePolicyEvaluation {
  private final StandardModel standardModel;
  private final ActionValueAdapter actionValueAdapter;
  private final Policy policy;
  private final Scalar gamma;
  private DiscreteVs vs_new;
  private DiscreteVs vs_old;
  private int iterations = 0;
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
  public IterativePolicyEvaluation( //
      StandardModel standardModel, Policy policy) {
    this.standardModel = standardModel;
    actionValueAdapter = new ActionValueAdapter(standardModel);
    this.policy = policy;
    this.gamma = standardModel.gamma();
    vs_new = DiscreteVs.build(standardModel.states());
  }

  /** @param threshold
   * @return */
  public void until(Scalar threshold) {
    until(threshold, Integer.MAX_VALUE);
  }

  public void until(Scalar threshold, int flips) {
    Scalar past = null;
    Timing timing = Timing.started();
    while (true) {
      step();
      Scalar delta = DiscreteValueFunctions.distance(vs_new, vs_old);
      if (3e9 < timing.nanoSeconds())
        System.out.println(past + " -> " + delta + " " + alternate);
      if (Objects.nonNull(past) && Scalars.lessThan(past, delta))
        if (flips < ++alternate) {
          System.out.println("give up at " + past + " -> " + delta);
          break;
        }
      past = delta;
      if (Scalars.lessThan(delta, threshold))
        break;
    }
  }

  public void step() {
    vs_old = vs_new.copy();
    VsInterface discounted = vs_new.discounted(gamma);
    vs_new = vs_new.create(standardModel.states().stream() //
        .parallel() //
        .map(state -> jacobiAdd(state, discounted)));
    ++iterations;
  }

  // helper function
  private Scalar jacobiAdd(Tensor state, VsInterface gvalues) {
    return standardModel.actions(state).stream() //
        .map(action -> policy.probability(state, action).multiply( //
            actionValueAdapter.qsa(state, action, gvalues))) //
        .reduce(Scalar::add) //
        .orElseThrow();
  }

  public DiscreteVs vs() {
    return vs_new;
  }

  public int iterations() {
    return iterations;
  }
}
