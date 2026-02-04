// code by jph
package ch.alpine.subare.book.ch06.maxbias;

import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.util.DiscreteVs;

/** the code produces the correct state value function
 *
 * Q-function
 * 
 * {0, 0} 0
 * {1, 0} -0.1
 * {1, 1} -0.1
 * {1, 2} -0.1
 * {1, 3} -0.1
 * {1, 4} -0.1
 * {2, -1} -0.1
 * {2, 1} 0
 * {3, 0} 0
 * 
 * V-function
 * 
 * 0 0
 * 1 -0.1
 * 2 0
 * 3 0 */
enum AVI_Maxbias {
  ;
  static void main() {
    Maxbias maxbias = new Maxbias(5);
    DiscreteQsa qsa = MaxbiasHelper.getOptimalQsa(maxbias);
    DiscreteUtils.print(qsa);
    System.out.println("---");
    DiscreteVs vs = DiscreteUtils.createVs(maxbias, qsa);
    DiscreteUtils.print(vs);
  }
}
