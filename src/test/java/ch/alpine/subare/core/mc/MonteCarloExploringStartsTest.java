// code by jph
package ch.alpine.subare.core.mc;

import ch.alpine.subare.core.MonteCarloInterface;
import ch.alpine.subare.core.Policy;
import ch.alpine.subare.core.adapter.SimpleTestModel;
import ch.alpine.subare.core.adapter.SimpleTestModels;
import ch.alpine.subare.core.util.ExploringStarts;
import ch.alpine.subare.core.util.PolicyType;
import junit.framework.TestCase;

public class MonteCarloExploringStartsTest extends TestCase {
  public void testSimple() {
    MonteCarloInterface monteCarloInterface = SimpleTestModel.INSTANCE;
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(monteCarloInterface);
    Policy policy = PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, mces.qsa(), mces.sac());
    ExploringStarts.batch(monteCarloInterface, policy, mces);
    // DiscreteUtils.print(mces.qsa());
    SimpleTestModels._checkExact(mces.qsa());
  }
}
