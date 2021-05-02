// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.alpine.tensor.RationalScalar;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.ethz.idsc.subare.core.DiscreteModel;
import junit.framework.TestCase;

public class LossTest extends TestCase {
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

  public void testAccumulation0() {
    DiscreteModel discreteModel = create14();
    DiscreteQsa ref = DiscreteQsa.build(discreteModel).create(Tensors.vector(0, 0, 0, 0).stream());
    DiscreteQsa qsa = DiscreteQsa.build(discreteModel).create(Tensors.vector(0, 0, 0, 0).stream());
    Scalar scalar = Loss.accumulation(discreteModel, ref, qsa);
    assertEquals(scalar, RealScalar.ZERO);
  }

  public void testAccumulation1() {
    DiscreteModel discreteModel = create14();
    DiscreteQsa ref = DiscreteQsa.build(discreteModel).create(Tensors.vector(0, 0, 1, 0).stream());
    DiscreteQsa qsa = DiscreteQsa.build(discreteModel).create(Tensors.vector(1, 0, 2, -2).stream());
    Scalar scalar = Loss.accumulation(discreteModel, ref, qsa);
    assertEquals(scalar, RealScalar.ZERO);
  }

  public void testAccumulation2() {
    DiscreteModel discreteModel = create14();
    DiscreteQsa ref = DiscreteQsa.build(discreteModel).create(Tensors.vector(0, 0, 1, 0).stream());
    DiscreteQsa qsa = DiscreteQsa.build(discreteModel).create(Tensors.vector(2, 0, 2, -2).stream());
    Scalar scalar = Loss.accumulation(discreteModel, ref, qsa);
    assertEquals(scalar, RationalScalar.HALF);
  }

  public void testAccumulation3() {
    DiscreteModel discreteModel = create14();
    DiscreteQsa ref = DiscreteQsa.build(discreteModel).create(Tensors.vector(0, 0, 1, 0).stream());
    DiscreteQsa qsa = DiscreteQsa.build(discreteModel).create(Tensors.vector(2, 2, 1.5, 2).stream());
    Scalar scalar = Loss.accumulation(discreteModel, ref, qsa);
    assertEquals(scalar, RealScalar.ONE);
  }
}
