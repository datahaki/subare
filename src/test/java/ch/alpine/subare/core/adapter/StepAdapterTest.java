// code by jph
package ch.alpine.subare.core.adapter;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.util.AssertFail;
import ch.alpine.tensor.Tensors;

public class StepAdapterTest {
  @Test
  public void testSimple() {
    AssertFail.of(() -> new StepAdapter(Tensors.empty(), Tensors.empty(), null, Tensors.empty()));
  }
}
