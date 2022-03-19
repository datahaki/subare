// code by jph
package ch.alpine.subare.core.adapter;

import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.Tensors;

public class StepAdapterTest {
  @Test
  public void testSimple() {
    assertThrows(Exception.class, () -> new StepAdapter(Tensors.empty(), Tensors.empty(), null, Tensors.empty()));
  }
}
