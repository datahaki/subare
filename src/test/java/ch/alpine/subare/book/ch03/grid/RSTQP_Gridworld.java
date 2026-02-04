// code by jph
package ch.alpine.subare.book.ch03.grid;

import ch.alpine.subare.alg.ActionValueIterations;
import ch.alpine.subare.alg.Random1StepTabularQPlanning;
import ch.alpine.subare.api.StepDigest;
import ch.alpine.subare.util.ConstantLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.TabularSteps;
import ch.alpine.tensor.RealScalar;

/* package */ enum RSTQP_Gridworld {
  ;
  static void main() {
    Gridworld gridworld = new Gridworld();
    DiscreteQsa ref = ActionValueIterations.solve(gridworld, RealScalar.of(0.0001));
    DiscreteQsa qsa = DiscreteQsa.build(gridworld);
    StepDigest stepDigest = Random1StepTabularQPlanning.of( //
        gridworld, qsa, ConstantLearningRate.one());
    for (int index = 0; index < 20; ++index) {
      Infoline infoline = Infoline.print(gridworld, index, ref, qsa);
      TabularSteps.batch(gridworld, gridworld, stepDigest);
      if (infoline.isLossfree())
        break;
    }
  }
}
