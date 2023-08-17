// code by jph
package ch.alpine.subare.td;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.api.LearningRate;
import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.util.DefaultLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.PolicyBase;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.subare.util.SimpleTestModel;
import ch.alpine.subare.util.SimpleTestModels;
import ch.alpine.tensor.RealScalar;

class PrioritizedSweepingTest {
  @Test
  void testSimple() {
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
