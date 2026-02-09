// code by jph
package ch.alpine.subare.alg;

import java.util.Objects;

import ch.alpine.subare.api.ActionValueInterface;
import ch.alpine.subare.api.DiscreteModel;
import ch.alpine.subare.api.DiscreteQsaSupplier;
import ch.alpine.subare.api.QsaInterface;
import ch.alpine.subare.api.StandardModel;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteValueFunctions;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.qty.Quantity;
import ch.alpine.tensor.qty.Timing;
import ch.alpine.tensor.red.Max;
import ch.alpine.tensor.sca.N;
import ch.alpine.tensor.sca.Sign;

/** action value iteration: "policy evaluation is stopped after just one sweep"
 * 
 * Exercise 4.10 on p.91
 * 
 * parallel implementation
 * initial values are set to zeros
 * Jacobi style, i.e. updates take effect only in the next iteration */
public class ActionValueIteration implements DiscreteQsaSupplier {
  /** @param standardModel */
  public static ActionValueIteration of(StandardModel standardModel) {
    return of(standardModel, standardModel);
  }

  public static ActionValueIteration of(StandardModel standardModel, DiscreteQsa qsa_new) {
    return new ActionValueIteration(standardModel, standardModel, qsa_new);
  }

  /** @param discreteModel
   * @param actionValueInterface */
  public static ActionValueIteration of(DiscreteModel discreteModel, ActionValueInterface actionValueInterface) {
    return new ActionValueIteration(discreteModel, actionValueInterface, DiscreteQsa.build(discreteModel));
  }

  // ---
  private final DiscreteModel discreteModel;
  private final ActionValueInterface actionValueInterface;
  private Scalar gamma;
  private DiscreteQsa qsa_new;
  private QsaInterface qsa_old;
  private int iterations = 0;
  private int alternate = 0;

  private ActionValueIteration( //
      DiscreteModel discreteModel, ActionValueInterface actionValueInterface, DiscreteQsa qsa_new) {
    this.discreteModel = discreteModel;
    this.actionValueInterface = actionValueInterface;
    this.gamma = discreteModel.gamma();
    this.qsa_new = qsa_new;
    StaticHelper.assertConsistent(qsa_new.keys(), actionValueInterface);
  }

  /** state-action values are stored in numeric precision */
  public void setMachinePrecision() {
    gamma = N.DOUBLE.apply(gamma);
  }

  /** perform iteration until values don't change more than threshold
   * 
   * @param threshold positive */
  public void untilBelow(Scalar threshold) {
    untilBelow(threshold, Integer.MAX_VALUE);
  }

  private static final Scalar LIMIT = Quantity.of(3e9, "ns");

  public void untilBelow(Scalar threshold, int flips) {
    Sign.requirePositive(threshold);
    Scalar past = null;
    Timing timing = Timing.started();
    while (true) {
      step();
      final Scalar delta = DiscreteValueFunctions.distance(qsa_new, (DiscreteQsa) qsa_old);
      if (Scalars.lessThan(LIMIT, timing.nanoSeconds())) // print info if iteration takes longer than 3 seconds
        IO.println(past + " -> " + delta + " " + alternate);
      if (Objects.nonNull(past) && Scalars.lessThan(past, delta))
        if (flips < ++alternate) {
          IO.println("give up at " + past + " -> " + delta);
          break;
        }
      past = delta;
      // TODO SUBARE consider changing to lessEquals (requires renaming of API functions)
      if (Scalars.lessThan(delta, threshold))
        break;
    }
  }

  /** perform one step of the iteration
   * 
   * @return */
  public void step() {
    qsa_old = qsa_new.copy();
    qsa_new = qsa_new.create(qsa_new.keys().stream() //
        .parallel() //
        .map(pair -> jacobiMax(pair.get(0), pair.get(1))));
    ++iterations;
  }

  // helper function
  private Scalar jacobiMax(Tensor state, Tensor action) {
    Scalar ersa = actionValueInterface.expectedReward(state, action);
    Scalar eqsa = ersa.zero();
    for (Tensor next : actionValueInterface.transitions(state, action)) {
      Scalar prob = actionValueInterface.transitionProbability(state, action, next);
      Scalar max = discreteModel.actions(next).stream() //
          .map(actionN -> qsa_new.value(next, actionN)) //
          .reduce(Max::of) //
          .orElseThrow();
      eqsa = eqsa.add(prob.multiply(max));
    }
    return ersa.add(gamma.multiply(eqsa));
  }

  @Override
  public DiscreteQsa qsa() {
    return qsa_new;
  }

  public int iterations() {
    return iterations;
  }
}
