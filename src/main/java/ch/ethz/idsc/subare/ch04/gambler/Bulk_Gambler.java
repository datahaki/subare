// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import java.awt.Point;

import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.LearningCompetition;
import ch.ethz.idsc.subare.core.util.LearningContender;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;

/** Sarsa applied to gambler for different learning rate parameters */
class Bulk_Gambler {
  public static void main(String[] args) throws Exception {
    Gambler gambler = new Gambler(20, RationalScalar.of(4, 10)); // 20, 4/10
    final DiscreteQsa ref = GamblerHelper.getOptimalQsa(gambler); // true q-function, for error measurement
    // ---
    SarsaType sarsaType = SarsaType.original;
    final Scalar errorcap = RealScalar.of(15); // 15
    final Scalar losscap = RealScalar.of(.25); // .5
    final Tensor epsilon = Subdivide.of(.2, .01, 100); // .2, .6
    int x = 0;
    LearningCompetition learningCompetition = new LearningCompetition( //
        ref, "gambler_Q_" + sarsaType.name() + "_E" + epsilon.Get(0), epsilon, errorcap, losscap);
    learningCompetition.NSTEP = 1;
    learningCompetition.MAGNIFY = 5;
    for (Tensor factor : Subdivide.of(.1, 10, 20)) { // .5 16
      int y = 0;
      for (Tensor exponent : Subdivide.of(.51, 1.5, 13)) { // .51 2
        DiscreteQsa qsa = DiscreteQsa.build(gambler);
        Sarsa sarsa = sarsaType.supply(gambler, qsa, DefaultLearningRate.of(factor.Get(), exponent.Get()));
        LearningContender learningContender = LearningContender.sarsa(gambler, sarsa);
        learningCompetition.put(new Point(x, y), learningContender);
        ++y;
      }
      ++x;
    }
    // ---
    learningCompetition.doit();
  }
}
