// code by jph
package ch.alpine.subare.book.ch06.windy;

import ch.alpine.subare.alg.ValueIteration;
import ch.alpine.subare.api.EpisodeInterface;
import ch.alpine.subare.api.Policy;
import ch.alpine.subare.api.StepRecord;
import ch.alpine.subare.util.EpisodeKickoff;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;

/** reproduces Figure 6.4 on p.139 */
enum VI_Windygrid {
  ;
  public static void simulate(Windygrid windygrid) {
    ValueIteration vi = new ValueIteration(windygrid, windygrid);
    vi.untilBelow(RealScalar.of(.001));
    final Tensor values = vi.vs().values();
    System.out.println("iterations=" + vi.iterations());
    System.out.println(values);
    Policy policy = PolicyType.GREEDY.bestEquiprobable(windygrid, vi.vs(), null);
    EpisodeInterface episodeInterface = EpisodeKickoff.single(windygrid, policy);
    while (episodeInterface.hasNext()) {
      StepRecord stepInterface = episodeInterface.step();
      System.out.println(stepInterface.prevState() + " + " + stepInterface.action() + " -> " + stepInterface.nextState());
    }
  }

  static void main() {
    simulate(Windygrid.createFour()); // reaches in
    simulate(Windygrid.createKing()); // reaches in 7
  }
}
