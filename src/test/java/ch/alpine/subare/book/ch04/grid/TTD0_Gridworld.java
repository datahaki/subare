// code by jph
package ch.alpine.subare.book.ch04.grid;

import ch.alpine.subare.api.Policy;
import ch.alpine.subare.td.TabularTemporalDifference0;
import ch.alpine.subare.util.DefaultLearningRate;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.util.DiscreteVs;
import ch.alpine.subare.util.EquiprobablePolicy;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.tensor.sca.Round;

/** Example 4.1, p.82
 * 
 * {0, 0} 0
 * {0, 1} -9.00
 * {0, 2} -19.90
 * {0, 3} -20.52
 * {1, 0} -13.61
 * {1, 1} -16.52
 * {1, 2} -17.52
 * {1, 3} -19.84
 * {2, 0} -16.20
 * {2, 1} -17.77
 * {2, 2} -19.94
 * {2, 3} -11.45
 * {3, 0} -21.01
 * {3, 1} -19.68
 * {3, 2} -18.52
 * {3, 3} 0 */
enum TTD0_Gridworld {
  ;
  public static void main(String[] args) {
    Gridworld gridWorld = new Gridworld();
    DiscreteVs vs = DiscreteVs.build(gridWorld.states());
    TabularTemporalDifference0 ttd0 = new TabularTemporalDifference0( //
        vs, gridWorld.gamma(), DefaultLearningRate.of(3, .6), new DiscreteStateActionCounter());
    Policy policy = EquiprobablePolicy.create(gridWorld);
    for (int count = 0; count < 300; ++count)
      ExploringStarts.batch(gridWorld, policy, ttd0);
    DiscreteUtils.print(vs, Round._2);
  }
}
