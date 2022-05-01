// code by jph
package ch.alpine.subare.core.mc;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.core.MonteCarloInterface;
import ch.alpine.subare.core.Policy;
import ch.alpine.subare.core.adapter.SimpleTestModel;
import ch.alpine.subare.core.adapter.SimpleTestModels;
import ch.alpine.subare.core.util.ExploringStarts;
import ch.alpine.subare.core.util.PolicyType;

class MonteCarloExploringStartsTest {
  @Test
  public void testSimple() {
    MonteCarloInterface monteCarloInterface = SimpleTestModel.INSTANCE;
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(monteCarloInterface);
    Policy policy = PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, mces.qsa(), mces.sac());
    ExploringStarts.batch(monteCarloInterface, policy, mces);
    // DiscreteUtils.print(mces.qsa());
    SimpleTestModels._checkExact(mces.qsa());
  }
}
