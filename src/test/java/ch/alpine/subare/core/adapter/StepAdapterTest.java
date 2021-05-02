// code by jph
package ch.alpine.subare.core.adapter;

import ch.alpine.subare.util.AssertFail;
import ch.alpine.tensor.Tensors;
import junit.framework.TestCase;

public class StepAdapterTest extends TestCase {
  public void testSimple() {
    AssertFail.of(() -> new StepAdapter(Tensors.empty(), Tensors.empty(), null, Tensors.empty()));
  }
}
