// code by jph
package ch.alpine.subare.book.ch04.grid;

import java.awt.Point;

import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.td.Sarsa;
import ch.alpine.subare.td.SarsaType;
import ch.alpine.subare.util.DefaultLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.EGreedyPolicy;
import ch.alpine.subare.util.LearningCompetition;
import ch.alpine.subare.util.LearningContender;
import ch.alpine.subare.util.LinearExplorationRate;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Subdivide;

/** Sarsa applied to gambler for different learning rate parameters */
enum Bulk_Gridworld {
  ;
  static void handle(SarsaType sarsaType, int nstep) throws Exception {
    Gridworld gambler = new Gridworld(); // 20, 4/10
    final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gambler); // true q-function, for error measurement
    // ---
    final Scalar errorcap = RealScalar.of(20); // 15
    final Scalar losscap = RealScalar.of(5); // .5
    final Tensor epsilon = Subdivide.of(.1, .01, 100); // .2, .6
    int x = 0;
    LearningCompetition learningCompetition = new LearningCompetition( //
        ref, "gridworld_" + sarsaType.name() + "_E" + epsilon.Get(0) + "_N" + nstep, epsilon, errorcap, losscap);
    learningCompetition.nstep = nstep;
    learningCompetition.magnify = 5;
    learningCompetition.period = 100;
    for (Tensor factor : Subdivide.of(.1, 10, 10)) { // .5 16
      int y = 0;
      for (Tensor exponent : Subdivide.of(.51, 1.3, 10)) { // .51 for qlearning use upper bound == 2, else == 1
        DiscreteQsa qsa = DiscreteQsa.build(gambler);
        StateActionCounter sac = new DiscreteStateActionCounter();
        EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(gambler, qsa, sac);
        policy.setExplorationRate(LinearExplorationRate.of(100, 0.1, 0.01));
        Sarsa sarsa = sarsaType.sarsa(gambler, DefaultLearningRate.of((Scalar) factor, (Scalar) exponent), qsa, sac, policy);
        LearningContender learningContender = LearningContender.sarsa(gambler, sarsa);
        learningCompetition.put(new Point(x, y), learningContender);
        ++y;
      }
      ++x;
    }
    // ---
    learningCompetition.doit();
  }

  static void main() throws Exception {
    handle(SarsaType.QLEARNING, 1);
  }
}
