// code by jph
// inspired by Shangtong Zhang
package ch.alpine.subare.book.ch04.grid;

import ch.alpine.subare.alg.IterativePolicyEvaluation;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.util.EquiprobablePolicy;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.sca.Round;

/** determines value function for equiprobable "random" policy
 * 
 * Example 4.1, p.82
 * Figure 4.1, p.83
 * 
 * {0, 0} 0
 * {0, 1} -14.0
 * {0, 2} -20.0
 * {0, 3} -22.0
 * {1, 0} -14.0
 * {1, 1} -18.0
 * {1, 2} -20.0
 * {1, 3} -20.0
 * {2, 0} -20.0
 * {2, 1} -20.0
 * {2, 2} -18.0
 * {2, 3} -14.0
 * {3, 0} -22.0
 * {3, 1} -20.0
 * {3, 2} -14.0
 * {3, 3} 0 */
enum IPE_Gridworld {
  ;
  public static void main(String[] args) {
    Gridworld gridworld = new Gridworld();
    IterativePolicyEvaluation ipe = new IterativePolicyEvaluation( //
        gridworld, EquiprobablePolicy.create(gridworld));
    ipe.until(RealScalar.of(.0001));
    DiscreteUtils.print(ipe.vs(), Round._1);
  }
}
