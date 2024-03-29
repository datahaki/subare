// code by jz
package ch.alpine.subare.book.ch04.gambler;

import java.util.ArrayDeque;
import java.util.Deque;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.red.Min;

@Deprecated
class ExplorationRateDeque {
  private double epsilon;
  private final Deque<Scalar> errors = new ArrayDeque<>();

  public ExplorationRateDeque(double epsilon) {
    this.epsilon = epsilon;
  }

  void notifyError(Scalar error) {
    errors.add(error);
    if (errors.size() == 2) {
      Scalar error_prev = errors.poll(); // n-5
      Scalar error_min = errors.stream().reduce(Min::of).orElseThrow();
      if (Scalars.lessThan(error_prev, error_min)) {
        epsilon /= 2;
        System.out.println("Current epsilon: " + epsilon);
        errors.clear();
      }
    }
  }

  public Scalar getEpsilon() {
    return RealScalar.of(epsilon);
  }
}
