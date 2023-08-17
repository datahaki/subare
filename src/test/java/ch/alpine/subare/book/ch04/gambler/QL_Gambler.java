// code by jz
package ch.alpine.subare.book.ch04.gambler;

import java.util.concurrent.TimeUnit;

import ch.alpine.subare.api.EpisodeInterface;
import ch.alpine.subare.api.Policy;
import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.api.StepRecord;
import ch.alpine.subare.td.Sarsa;
import ch.alpine.subare.td.SarsaType;
import ch.alpine.subare.util.DefaultLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.util.EGreedyPolicy;
import ch.alpine.subare.util.EpisodeKickoff;
import ch.alpine.subare.util.EquiprobablePolicy;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.LearningRate;
import ch.alpine.subare.util.LinearExplorationRate;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.subare.util.gfx.StateActionRasters;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.AnimationWriter;
import ch.alpine.tensor.io.GifAnimationWriter;
import ch.alpine.tensor.sca.Round;

/** Q-Learning applied to gambler with adaptive learning rate */
/* package */ enum QL_Gambler {
  ;
  static void handle() throws Exception {
    GamblerModel gamblerModel = GamblerModel.createDefault();
    final DiscreteQsa ref = GamblerHelper.getOptimalQsa(gamblerModel);
    int batches = 100;
    Policy policy = EquiprobablePolicy.create(gamblerModel);
    DiscreteQsa qsa = DiscreteQsa.build(gamblerModel);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policyEGreedy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(gamblerModel, qsa, sac);
    policyEGreedy.setExplorationRate(LinearExplorationRate.of(batches, 0.1, 0.01));
    System.out.println(qsa.size());
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures("gambler_qsa_ql.gif"), 100, TimeUnit.MILLISECONDS)) {
      LearningRate learningRate = DefaultLearningRate.of(2, 0.51);
      Sarsa stepDigest = SarsaType.QLEARNING.sarsa(gamblerModel, learningRate, qsa, sac, policyEGreedy);
      for (int index = 0; index < batches; ++index) {
        Infoline.print(gamblerModel, index, ref, qsa);
        for (int count = 0; count < 1; ++count) {
          ExploringStarts.batch(gamblerModel, policy, 1, stepDigest);
        }
        animationWriter.write(StateActionRasters.qsaPolicyRef(new GamblerRaster(gamblerModel), qsa, ref));
      }
    }
    DiscreteUtils.print(qsa, Round._2);
    System.out.println("---");
    EpisodeInterface mce = EpisodeKickoff.single(gamblerModel, policy);
    while (mce.hasNext()) {
      StepRecord stepInterface = mce.step();
      Tensor state = stepInterface.prevState();
      System.out.println(state + " then " + stepInterface.action());
    }
  }

  public static void main(String[] args) throws Exception {
    handle();
  }
}
