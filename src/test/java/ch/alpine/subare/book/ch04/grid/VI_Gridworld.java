// code by jph
// inspired by Shangtong Zhang
package ch.alpine.subare.book.ch04.grid;

import ch.alpine.subare.alg.ValueIteration;
import ch.alpine.subare.api.Policy;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.util.DiscreteValueFunctions;
import ch.alpine.subare.util.Policies;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.subare.util.gfx.StateRasters;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.Export;

/** solving grid world
 * gives the value function for the optimal policy equivalent to
 * shortest path to terminal state
 *
 * Example 4.1, p.82
 * 
 * {0, 0} 0
 * {0, 1} -1
 * {0, 2} -2
 * {0, 3} -3
 * {1, 0} -1
 * {1, 1} -2
 * {1, 2} -3
 * {1, 3} -2
 * {2, 0} -2
 * {2, 1} -3
 * {2, 2} -2
 * {2, 3} -1
 * {3, 0} -3
 * {3, 1} -2
 * {3, 2} -1
 * {3, 3} 0 */
enum VI_Gridworld {
  ;
  static void main() throws Exception {
    Gridworld gridworld = new Gridworld();
    GridworldRaster gridworldStateRaster = new GridworldRaster(gridworld);
    ValueIteration vi = new ValueIteration(gridworld, gridworld);
    vi.untilBelow(RealScalar.of(.0001));
    DiscreteUtils.print(vi.vs());
    Policy policy = PolicyType.GREEDY.bestEquiprobable(gridworld, vi.vs(), null);
    Policies.print(policy, gridworld.states());
    Export.of(HomeDirectory.Pictures.resolve("gridworld_vs_vi.png"), //
        StateRasters.vs(gridworldStateRaster, DiscreteValueFunctions.rescaled(vi.vs())));
    // GridworldHelper.render(gridworld, vi.vs())
  }
}
