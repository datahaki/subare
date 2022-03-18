// code by jph
package ch.alpine.subare.core.td;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.core.StateActionCounter;
import ch.alpine.subare.core.adapter.SimpleTestModel;
import ch.alpine.subare.core.adapter.SimpleTestModels;
import ch.alpine.subare.core.util.DefaultLearningRate;
import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.subare.core.util.DiscreteStateActionCounter;
import ch.alpine.subare.core.util.ExploringStarts;
import ch.alpine.subare.core.util.LearningRate;
import ch.alpine.subare.core.util.PolicyBase;
import ch.alpine.subare.core.util.PolicyType;
import ch.alpine.tensor.RealScalar;

public class PrioritizedSweepingTest {
  @Test
  public void testSimple() {
    SimpleTestModel simpleTestModel = SimpleTestModel.INSTANCE;
    LearningRate learningRate = DefaultLearningRate.of(8, 2);
    DiscreteQsa qsa = DiscreteQsa.build(simpleTestModel, RealScalar.ZERO);
    StateActionCounter sac = new DiscreteStateActionCounter();
    PolicyBase policy = PolicyType.EGREEDY.bestEquiprobable(simpleTestModel, qsa, sac);
    Sarsa sarsa = SarsaType.ORIGINAL.sarsa(simpleTestModel, learningRate, qsa, sac, policy);
    PrioritizedSweeping ps = new PrioritizedSweeping(sarsa, 2, RealScalar.of(.1));
    ExploringStarts.batch(simpleTestModel, policy, ps);
    SimpleTestModels._checkExact(qsa);
  }
}
