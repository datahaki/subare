// code by jph
package ch.alpine.subare.ch04.gambler;

import ch.alpine.subare.core.EpisodeInterface;
import ch.alpine.subare.core.Policy;
import ch.alpine.subare.core.StepRecord;
import ch.alpine.subare.core.alg.ActionValueIterations;
import ch.alpine.subare.core.alg.ValueIteration;
import ch.alpine.subare.core.alg.ValueIterations;
import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.subare.core.util.DiscreteUtils;
import ch.alpine.subare.core.util.DiscreteVs;
import ch.alpine.subare.core.util.EpisodeKickoff;
import ch.alpine.subare.core.util.PolicyType;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.sca.Round;

/* package */ enum GamblerHelper {
  ;
  static DiscreteQsa getOptimalQsa(GamblerModel gamblerModel) {
    return ActionValueIterations.solve(gamblerModel, RealScalar.of(.0001));
  }

  public static DiscreteVs getOptimalVs(GamblerModel gamblerModel) {
    return ValueIterations.solve(gamblerModel, RealScalar.of(1e-10));
  }

  public static Policy getOptimalPolicy(GamblerModel gamblerModel) {
    // TODO SUBARE test for equality of policies from qsa and vs
    ValueIteration vi = new ValueIteration(gamblerModel, gamblerModel);
    vi.untilBelow(RealScalar.of(1e-10));
    return PolicyType.GREEDY.bestEquiprobable(gamblerModel, vi.vs(), null);
  }

  public static void play(GamblerModel gamblerModel, DiscreteQsa qsa) {
    DiscreteUtils.print(qsa, Round._2);
    System.out.println("---");
    Policy policy = PolicyType.GREEDY.bestEquiprobable(gamblerModel, qsa, null);
    EpisodeInterface mce = EpisodeKickoff.single(gamblerModel, policy, //
        gamblerModel.startStates().get(gamblerModel.startStates().length() / 2));
    while (mce.hasNext()) {
      StepRecord stepInterface = mce.step();
      Tensor state = stepInterface.prevState();
      System.out.println(state + " then " + stepInterface.action());
    }
  }
}
