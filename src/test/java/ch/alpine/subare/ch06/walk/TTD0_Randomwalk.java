// code by jph
package ch.alpine.subare.ch06.walk;

import ch.alpine.subare.core.api.Policy;
import ch.alpine.subare.core.api.VsInterface;
import ch.alpine.subare.core.td.TabularTemporalDifference0;
import ch.alpine.subare.core.util.DefaultLearningRate;
import ch.alpine.subare.core.util.DiscreteStateActionCounter;
import ch.alpine.subare.core.util.DiscreteUtils;
import ch.alpine.subare.core.util.DiscreteVs;
import ch.alpine.subare.core.util.EquiprobablePolicy;
import ch.alpine.subare.core.util.ExploringStarts;
import ch.alpine.subare.core.util.LinearApproximationVs;
import ch.alpine.subare.util.PolynomialBasis;
import ch.alpine.tensor.alg.Array;
import ch.alpine.tensor.sca.Clips;
import ch.alpine.tensor.sca.Round;

/** tabular temporal difference (0) to learn value of states
 * 
 * <pre>
 * 0 0
 * 1 0.10
 * 2 0.27
 * 3 0.47
 * 4 0.67
 * 5 0.90
 * 6 0
 * </pre> */
enum TTD0_Randomwalk {
  ;
  static void some(Randomwalk randomwalk, VsInterface vs) {
    TabularTemporalDifference0 ttd0 = new TabularTemporalDifference0( //
        vs, randomwalk.gamma(), DefaultLearningRate.of(3, .6), new DiscreteStateActionCounter());
    Policy policy = EquiprobablePolicy.create(randomwalk);
    for (int count = 0; count < 1000; ++count)
      ExploringStarts.batch(randomwalk, policy, ttd0);
    DiscreteUtils.print(randomwalk, vs, Round._2);
  }

  public static void main(String[] args) {
    Randomwalk randomwalk = new Randomwalk(5);
    some(randomwalk, DiscreteVs.build(randomwalk.states()));
    int order = 2;
    some(randomwalk, //
        LinearApproximationVs.create(new PolynomialBasis(order, Clips.positive(6)), Array.zeros(order)));
  }
}
