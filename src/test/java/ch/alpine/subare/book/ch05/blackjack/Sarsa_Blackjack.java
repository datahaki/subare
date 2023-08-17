// code by jph
package ch.alpine.subare.book.ch05.blackjack;

import java.util.concurrent.TimeUnit;

import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.td.Sarsa;
import ch.alpine.subare.td.SarsaType;
import ch.alpine.subare.util.DefaultLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.EGreedyPolicy;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.LinearExplorationRate;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Subdivide;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.AnimationWriter;
import ch.alpine.tensor.io.GifAnimationWriter;
import ch.alpine.tensor.sca.Round;

// TODO SUBARE this demo throws an exception
/* package */ enum Sarsa_Blackjack {
  ;
  static void handle(SarsaType sarsaType) throws Exception {
    System.out.println(sarsaType);
    final Blackjack blackjack = new Blackjack();
    int batches = 40;
    Tensor epsilon = Subdivide.of(.1, .01, batches); // only used in egreedy
    DiscreteQsa qsa = DiscreteQsa.build(blackjack);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(blackjack, qsa, sac);
    policy.setExplorationRate(LinearExplorationRate.of(batches, 0.1, 0.01));
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures("blackjack_qsa_" + sarsaType + ".gif"), 200, TimeUnit.MILLISECONDS)) {
      Sarsa sarsa = sarsaType.sarsa(blackjack, DefaultLearningRate.of(2, 0.6), qsa, sac, policy);
      for (int index = 0; index < batches; ++index) {
        // Scalar error = DiscreteQsas.distance(qsa, ref);
        System.out.println(index + " " + epsilon.Get(index).map(Round._2));
        // sarsa.supplyPolicy(() -> policy);
        for (int count = 0; count < 10; ++count)
          ExploringStarts.batch(blackjack, policy, sarsa);
        animationWriter.write(BlackjackHelper.joinAll(blackjack, qsa));
      }
    }
  }

  public static void main(String[] args) throws Exception {
    handle(SarsaType.QLEARNING);
  }
}
