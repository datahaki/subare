// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch04.gambler;

import java.io.IOException;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.alg.ValueIteration;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.core.util.Policies;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.io.Put;

/** Shangtong Zhang states that using double precision in python
 * "due to tie and precision, can't reproduce the optimal policy in book"
 * 
 * Unlike stated in the book, there is not a unique optimal policy but many
 * using symbolic expressions we can reproduce the policy in book and
 * all other optimal actions
 * 
 * chapter 4, example 3 */
class VI_Gambler {
  public static void main(String[] args) throws IOException {
    Gambler gambler = Gambler.createDefault();
    ValueIteration vi = new ValueIteration(gambler);
    vi.untilBelow(RealScalar.of(1e-10));
    Tensor values = vi.vs().values();
    System.out.println(values);
    Put.of(UserHome.file("ex403_values"), values);
    PolicyInterface policyInterface = GreedyPolicy.bestEquiprobable(gambler, vi.vs());
    Policies.print(policyInterface, gambler.states());
    // System.out.println(greedyPolicy.policy(RealScalar.of(49), RealScalar.of(1)));
    Tensor greedy = Policies.flatten(policyInterface, gambler.states());
    Put.of(UserHome.file("ex403_greedy"), greedy);
  }
}