// code by jph
package ch.ethz.idsc.subare.ch03.grid;

import ch.ethz.idsc.subare.core.alg.ActionValueIterations;
import ch.ethz.idsc.subare.core.alg.Random1StepTabularQPlanning;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.TabularSteps;
import ch.ethz.idsc.subare.core.util.TensorValuesUtils;
import ch.ethz.idsc.subare.util.Digits;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;

class RSTQP_Gridworld {
  public static void main(String[] args) {
    Gridworld gridworld = new Gridworld();
    DiscreteQsa ref = ActionValueIterations.solve(gridworld, DecimalScalar.of(0.0001));
    DiscreteQsa qsa = DiscreteQsa.build(gridworld);
    Random1StepTabularQPlanning rstqp = new Random1StepTabularQPlanning(gridworld, qsa);
    rstqp.setLearningRate(RealScalar.of(1.));
    for (int index = 0; index < 40; ++index) {
      TabularSteps.batch(gridworld, gridworld, rstqp);
      Scalar error = TensorValuesUtils.distance(ref, qsa);
      System.out.println(index + " " + error.map(Digits._1));
    }
  }
}
