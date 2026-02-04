// code by jph
package ch.alpine.subare.book.ch08.maze;

import ch.alpine.subare.math.Index;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteValueFunctions;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.nrm.Vector1Norm;
import ch.alpine.tensor.sca.pow.Power;

enum DynamazeHeuristic {
  ;
  public static DiscreteQsa create(Dynamaze dynamaze) {
    Index terminalIndex = dynamaze.terminalIndex();
    if (terminalIndex.size() != 1)
      throw new RuntimeException("not yet implemented");
    Tensor endPos = terminalIndex.get(0); // for maze2 == {8, 0}
    DiscreteQsa qsa = DiscreteQsa.build(dynamaze);
    for (Tensor key : qsa.keys()) {
      final Tensor state = key.get(0);
      final Tensor action = key.get(1);
      Scalar dist = Vector1Norm.between(state.add(action), endPos);
      // Scalar dist = Norm._1.ofVector(state.subtract(endPos));
      Scalar value = Power.of(dynamaze.gamma(), dist);
      qsa.assign(state, action, value);
    }
    return qsa;
  }

  public static void demo(Dynamaze dynamaze) {
    DiscreteQsa nul = DiscreteQsa.build(dynamaze);
    DiscreteQsa est = create(dynamaze);
    DiscreteQsa qsa = DynamazeHelper.getOptimalQsa(dynamaze);
    System.out.println("diff to zero      = " + DiscreteValueFunctions.distance(qsa, nul));
    System.out.println("diff to heuristic = " + DiscreteValueFunctions.distance(qsa, est));
  }

  static void main() {
    demo(DynamazeHelper.original("maze2"));
    demo(DynamazeHelper.create5(2));
  }
}
