// code by jph
// inspired by Shangtong Zhang
package ch.alpine.subare.book.ch06.cliff;

import ch.alpine.subare.alg.ValueIteration;
import ch.alpine.subare.api.EpisodeInterface;
import ch.alpine.subare.api.Policy;
import ch.alpine.subare.api.StepRecord;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.util.DiscreteValueFunctions;
import ch.alpine.subare.util.DiscreteVs;
import ch.alpine.subare.util.EpisodeKickoff;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.subare.util.gfx.StateRasters;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.Export;

/** value iteration for cliffwalk */
enum VI_Cliffwalk {
  ;
  static void main() throws Exception {
    Cliffwalk cliffwalk = new Cliffwalk(12, 4);
    CliffwalkRaster cliffwalkRaster = new CliffwalkRaster(cliffwalk);
    DiscreteQsa ref = CliffwalkHelper.getOptimalQsa(cliffwalk);
    ValueIteration vi = new ValueIteration(cliffwalk, cliffwalk);
    vi.untilBelow(RealScalar.of(.0001));
    DiscreteVs vs = vi.vs();
    DiscreteVs vr = DiscreteUtils.createVs(cliffwalk, ref);
    Scalar error = DiscreteValueFunctions.distance(vs, vr);
    System.out.println("error=" + error);
    Export.of(HomeDirectory.Pictures("cliffwalk_qsa_vi.png"), //
        StateRasters.vs_rescale(cliffwalkRaster, vi.vs()));
    // GreedyPolicy greedyPolicy = GreedyPolicy.bestEquiprobableGreedy(cliffWalk, values);
    // greedyPolicy.print(cliffWalk.states());
    // Index statesIndex = Index.build(cliffWalk.states());
    // for (int stateI = 0; stateI < statesIndex.size(); ++stateI) {
    // Tensor state = statesIndex.get(stateI);
    // System.out.println(state + " " + values.get(stateI).map(ROUND));
    // }
    Policy policy = PolicyType.GREEDY.bestEquiprobable(cliffwalk, ref, null);
    EpisodeInterface mce = EpisodeKickoff.single(cliffwalk, policy);
    while (mce.hasNext()) {
      StepRecord stepRecord = mce.step();
      Tensor state = stepRecord.prevState();
      System.out.println(state + " then " + stepRecord.action());
    }
  }
}
