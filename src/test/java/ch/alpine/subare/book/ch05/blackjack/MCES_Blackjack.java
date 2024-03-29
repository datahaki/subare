// code by jph
package ch.alpine.subare.book.ch05.blackjack;

import java.util.concurrent.TimeUnit;

import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.mc.MonteCarloExploringStarts;
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

/** Example 5.3 p.108: Solving Blackjack
 * Figure 5.3 p.108
 * 
 * finding optimal policy to stay or hit */
enum MCES_Blackjack {
  ;
  public static void main(String[] args) throws Exception {
    Blackjack blackjack = new Blackjack();
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(blackjack);
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures("blackjack_mces.gif"), 250, TimeUnit.MILLISECONDS)) {
      int batches = 10; // 40
      Tensor epsilon = Subdivide.of(.2, .05, batches);
      StateActionCounter sac = new DiscreteStateActionCounter();
      EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(blackjack, mces.qsa(), sac);
      policy.setExplorationRate(LinearExplorationRate.of(batches, 0.1, 0.01));
      int episodes = 0;
      for (int index = 0; index < batches; ++index) {
        System.out.println(index + " " + epsilon.Get(index));
        for (int count = 0; count < batches; ++count) {
          episodes += ExploringStarts.batchWithReplay(blackjack, policy, mces);
        }
        animationWriter.write(BlackjackHelper.joinAll(blackjack, mces.qsa()));
        System.out.println(episodes);
      }
    }
  }
}
