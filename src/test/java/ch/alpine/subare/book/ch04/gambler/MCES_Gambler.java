// code by jph
package ch.alpine.subare.book.ch04.gambler;

import java.util.concurrent.TimeUnit;

import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.mc.MonteCarloExploringStarts;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.util.DiscreteVs;
import ch.alpine.subare.util.EGreedyPolicy;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.subare.util.gfx.StateActionRasters;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.AnimationWriter;
import ch.alpine.tensor.io.GifAnimationWriter;
import ch.alpine.tensor.sca.Round;

/* package */ enum MCES_Gambler {
  ;
  public static void main(String[] args) throws Exception {
    GamblerModel gambler = GamblerModel.createDefault();
    GamblerRaster gamblerRaster = new GamblerRaster(gambler);
    final DiscreteQsa ref = GamblerHelper.getOptimalQsa(gambler);
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(gambler);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(gambler, mces.qsa(), sac);
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures("gambler_qsa_mces.gif"), 200, TimeUnit.MILLISECONDS)) {
      int batches = 20;
      for (int index = 0; index < batches; ++index) {
        Infoline.print(gambler, index, ref, mces.qsa());
        ExploringStarts.batch(gambler, policy, mces);
        animationWriter.write(StateActionRasters.qsaPolicyRef(gamblerRaster, mces.qsa(), ref));
      }
    }
    System.out.println("done");
    DiscreteVs discreteVs = DiscreteUtils.createVs(gambler, mces.qsa());
    DiscreteUtils.print(discreteVs, Round._2);
  }
}
