// code by jph
package ch.alpine.subare.book.ch06.maxbias;

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
enum Bulk_Maxbias {
  ;
  static void handle(SarsaType sarsaType, int nstep) throws Exception {
    Maxbias maxbias = new Maxbias(1); // 20, 4/10
    final DiscreteQsa ref = MaxbiasHelper.getOptimalQsa(maxbias); // true q-function, for error measurement
    // ---
    final Scalar errorcap = RealScalar.of(.5); // 15
    final Scalar losscap = RealScalar.of(.5); // .5
    final Tensor epsilon = Subdivide.of(.2, .01, 100); // .2, .6
    int x = 0;
    LearningCompetition learningCompetition = new LearningCompetition( //
        ref, "maxbias_" + sarsaType.name() + "_E" + epsilon.Get(0) + "_N" + nstep, epsilon, errorcap, losscap);
    learningCompetition.nstep = nstep;
    learningCompetition.magnify = 5;
    learningCompetition.period = 100;
    for (Tensor factor : Subdivide.of(.1, 10, 20)) { // .5 16
      int y = 0;
      for (Tensor exponent : Subdivide.of(.51, 2, 10)) { // .51 2
        DiscreteQsa qsa = DiscreteQsa.build(maxbias);
        StateActionCounter sac = new DiscreteStateActionCounter();
        EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(maxbias, qsa, sac);
        policy.setExplorationRate(LinearExplorationRate.of(100, 0.2, 0.01));
        Sarsa sarsa = sarsaType.sarsa(maxbias, DefaultLearningRate.of((Scalar) factor, (Scalar) exponent), qsa, sac, policy);
        LearningContender learningContender = LearningContender.sarsa(maxbias, sarsa);
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
