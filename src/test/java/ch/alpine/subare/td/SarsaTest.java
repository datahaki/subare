// code by jph
package ch.alpine.subare.td;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.api.LearningRate;
import ch.alpine.subare.api.MonteCarloInterface;
import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.util.ConstantLearningRate;
import ch.alpine.subare.util.DefaultLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.PolicyBase;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.subare.util.SimpleTestModel;
import ch.alpine.subare.util.SimpleTestModels;
import ch.alpine.subare.util.StateAction;
import ch.alpine.tensor.RealScalar;

class SarsaTest {
  @Test
  void testConstantOneExact() {
    for (SarsaType sarsaType : SarsaType.values()) {
      MonteCarloInterface monteCarloInterface = SimpleTestModel.INSTANCE;
      LearningRate learningRate = ConstantLearningRate.one();
      DiscreteQsa qsa = DiscreteQsa.build(monteCarloInterface);
      StateActionCounter sac = new DiscreteStateActionCounter();
      PolicyBase policy = PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, qsa, sac);
      Sarsa sarsa = sarsaType.sarsa(monteCarloInterface, learningRate, qsa, sac, policy);
      assertFalse(sarsa.sac().isEncountered(StateAction.key(RealScalar.ZERO, RealScalar.ONE)));
      ExploringStarts.batch(monteCarloInterface, policy, 2, sarsa); // nstep > 1 required
      // DiscreteUtils.print(qsa);
      SimpleTestModels._checkExact(qsa);
      assertTrue(sarsa.sac().isEncountered(StateAction.key(RealScalar.ZERO, RealScalar.ONE)));
    }
  }

  @Test
  void testConstantNonOneExact() {
    for (SarsaType sarsaType : SarsaType.values()) {
      MonteCarloInterface monteCarloInterface = SimpleTestModel.INSTANCE;
      DiscreteQsa qsa = DiscreteQsa.build(monteCarloInterface);
      LearningRate learningRate = ConstantLearningRate.of(RealScalar.of(.8));
      StateActionCounter sac = new DiscreteStateActionCounter();
      PolicyBase policy = PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, qsa, sac);
      Sarsa sarsa = sarsaType.sarsa(monteCarloInterface, learningRate, qsa, sac, policy);
      assertFalse(sarsa.sac().isEncountered(StateAction.key(RealScalar.ZERO, RealScalar.ONE)));
      ExploringStarts.batch(monteCarloInterface, policy, 2, sarsa); // nstep > 1 required
      // DiscreteUtils.print(qsa);
      SimpleTestModels._checkExact(qsa);
      assertTrue(sarsa.sac().isEncountered(StateAction.key(RealScalar.ZERO, RealScalar.ONE)));
    }
  }

  @Test
  void testDefaultExact() {
    for (SarsaType sarsaType : SarsaType.values()) {
      MonteCarloInterface monteCarloInterface = SimpleTestModel.INSTANCE;
      LearningRate learningRate = DefaultLearningRate.of(1.5, 0.6);
      DiscreteQsa qsa = DiscreteQsa.build(monteCarloInterface);
      StateActionCounter sac = new DiscreteStateActionCounter();
      PolicyBase policy = PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, qsa, sac);
      Sarsa sarsa = sarsaType.sarsa(monteCarloInterface, learningRate, qsa, sac, policy);
      assertFalse(sarsa.sac().isEncountered(StateAction.key(RealScalar.ZERO, RealScalar.ONE)));
      ExploringStarts.batch(monteCarloInterface, policy, 2, sarsa); // nstep > 1 required
      // DiscreteUtils.print(qsa);
      SimpleTestModels._checkExact(qsa);
      assertTrue(sarsa.sac().isEncountered(StateAction.key(RealScalar.ZERO, RealScalar.ONE)));
    }
  }
}
