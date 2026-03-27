// code by jph
package ch.alpine.subare.alg;

import ch.alpine.subare.mod.ActionValueInterface;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Throw;
import ch.alpine.tensor.mat.Tolerance;
import ch.alpine.tensor.sca.Sign;

enum StaticHelper {
  ;
  // test that probabilities add up to 1
  static void assertConsistent(Tensor keys, ActionValueInterface actionValueInterface) {
    keys.stream().parallel() //
        .forEach(pair -> _isConsistent(actionValueInterface, pair.get(0), pair.get(1)));
  }

  private static void _isConsistent(ActionValueInterface actionValueInterface, Tensor state, Tensor action) {
    Scalar norm = actionValueInterface.transitions(state, action).stream() //
        .map(next -> actionValueInterface.transitionProbability(state, action, next)) //
        .map(Sign::requirePositiveOrZero) //
        .reduce(Scalar::add) //
        .orElseThrow();
    if (!Tolerance.CHOP.isClose(norm, RealScalar.ONE)) {
      IO.println("state =" + state);
      IO.println("action=" + action);
      actionValueInterface.transitions(state, action).forEach(next -> {
        Scalar prob = actionValueInterface.transitionProbability(state, action, next);
        IO.println(next + " " + prob);
      });
      throw new Throw(norm, state, action); // probabilities have to sum up to 1
    }
  }
}
