// code by jph
package ch.alpine.subare.util;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.api.DiscreteModel;
import ch.alpine.tensor.RationalScalar;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;

class LossTest {
  static DiscreteModel create14() {
    return new DiscreteModel() {
      @Override
      public Scalar gamma() {
        return null;
      }

      @Override
      public Tensor states() {
        return Tensors.vector(0);
      }

      @Override
      public Tensor actions(Tensor state) {
        return Tensors.vector(0, 1, 2, 3);
      }
    };
  }

  @Test
  void testAccumulation0() {
    DiscreteModel discreteModel = create14();
    DiscreteQsa ref = DiscreteQsa.build(discreteModel).create(Tensors.vector(0, 0, 0, 0).stream());
    DiscreteQsa qsa = DiscreteQsa.build(discreteModel).create(Tensors.vector(0, 0, 0, 0).stream());
    Scalar scalar = Loss.accumulation(discreteModel, ref, qsa);
    assertEquals(scalar, RealScalar.ZERO);
  }

  @Test
  void testAccumulation1() {
    DiscreteModel discreteModel = create14();
    DiscreteQsa ref = DiscreteQsa.build(discreteModel).create(Tensors.vector(0, 0, 1, 0).stream());
    DiscreteQsa qsa = DiscreteQsa.build(discreteModel).create(Tensors.vector(1, 0, 2, -2).stream());
    Scalar scalar = Loss.accumulation(discreteModel, ref, qsa);
    assertEquals(scalar, RealScalar.ZERO);
  }

  @Test
  void testAccumulation2() {
    DiscreteModel discreteModel = create14();
    DiscreteQsa ref = DiscreteQsa.build(discreteModel).create(Tensors.vector(0, 0, 1, 0).stream());
    DiscreteQsa qsa = DiscreteQsa.build(discreteModel).create(Tensors.vector(2, 0, 2, -2).stream());
    Scalar scalar = Loss.accumulation(discreteModel, ref, qsa);
    assertEquals(scalar, RationalScalar.HALF);
  }

  @Test
  void testAccumulation3() {
    DiscreteModel discreteModel = create14();
    DiscreteQsa ref = DiscreteQsa.build(discreteModel).create(Tensors.vector(0, 0, 1, 0).stream());
    DiscreteQsa qsa = DiscreteQsa.build(discreteModel).create(Tensors.vector(2, 2, 1.5, 2).stream());
    Scalar scalar = Loss.accumulation(discreteModel, ref, qsa);
    assertEquals(scalar, RealScalar.ONE);
  }
}
