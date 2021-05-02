// code by jph
package ch.ethz.idsc.subare.ch05.blackjack;

import java.util.concurrent.TimeUnit;

import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.AnimationWriter;
import ch.alpine.tensor.io.GifAnimationWriter;
import ch.ethz.idsc.subare.core.alg.Random1StepTabularQPlanning;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.TabularSteps;

/** finding optimal policy to stay or hit
 * 
 * Random1StepTabularQPlanning does not seem to work on blackjack */
enum RSTQP_Blackjack {
  ;
  public static void main(String[] args) throws Exception {
    Blackjack blackjack = new Blackjack();
    DiscreteQsa qsa = DiscreteQsa.build(blackjack);
    Random1StepTabularQPlanning rstqp = Random1StepTabularQPlanning.of( //
        blackjack, qsa, DefaultLearningRate.of(5, 0.51));
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures("blackjack_rstqp.gif"), 250, TimeUnit.MILLISECONDS)) {
      int batches = 60;
      for (int index = 0; index < batches; ++index) {
        for (int count = 0; count < 100; ++count)
          TabularSteps.batch(blackjack, blackjack, rstqp);
        animationWriter.write(BlackjackHelper.joinAll(blackjack, qsa));
      }
    }
  }
}
