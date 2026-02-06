// code by jph
package ch.alpine.subare.book.ch05.blackjack;

import java.util.concurrent.TimeUnit;

import ch.alpine.subare.alg.ActionValueIteration;
import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.mc.MonteCarloExploringStarts;
import ch.alpine.subare.util.ActionValueStatistics;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.EGreedyPolicy;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.LinearExplorationRate;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Join;
import ch.alpine.tensor.alg.Subdivide;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.AnimationWriter;
import ch.alpine.tensor.io.GifAnimationWriter;

/** finding optimal policy to stay or hit
 * 
 * Figure 5.3 p.108 */
enum AVS_Blackjack {
  ;
  static void main() throws Exception {
    Blackjack blackjack = new Blackjack();
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(blackjack);
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures.resolve("blackjack_avs.gif"), 250, TimeUnit.MILLISECONDS)) {
      int batches = 3; // 40
      Tensor epsilon = Subdivide.of(.2, .05, batches);
      StateActionCounter sac = new DiscreteStateActionCounter();
      EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(blackjack, mces.qsa(), sac);
      policy.setExplorationRate(LinearExplorationRate.of(batches, 0.1, 0.01));
      int episodes = 0;
      ActionValueStatistics avs = new ActionValueStatistics(blackjack);
      for (int index = 0; index < batches; ++index) {
        System.out.println(index + " " + epsilon.Get(index));
        for (int count = 0; count < batches; ++count) {
          episodes += ExploringStarts.batchWithReplay(blackjack, policy, mces, avs);
        }
        ActionValueIteration avi = ActionValueIteration.of(blackjack, avs);
        avi.untilBelow(RealScalar.of(.0001), 3);
        animationWriter.write( //
            Join.of( //
                BlackjackHelper.joinAll(blackjack, mces.qsa()), //
                BlackjackHelper.joinAll(blackjack, avi.qsa())));
        System.out.println(episodes + " " + avs.coverage());
      }
    }
  }
}
