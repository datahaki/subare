// code by jph
package ch.alpine.subare.net.lay;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.net.DropoutLayers;
import ch.alpine.subare.net.NetLayer;
import ch.alpine.tensor.Rational;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.ext.Serialization;
import ch.alpine.tensor.mat.Tolerance;
import ch.alpine.tensor.nrm.FrobeniusNorm;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;
import ch.alpine.tensor.red.Mean;
import ch.alpine.tensor.sca.Chop;

class DropoutLayerTest {
  @Test
  void test() {
    NetLayer dropoutLayer = DropoutLayer.of(0.3);
    Tensor X = RandomVariate.of(NormalDistribution.standard(), 3, 4, 5);
    Tensor Y = dropoutLayer.forwardBatch(X);
    Tolerance.CHOP.requireClose(X, Y);
    DropoutLayers.setEnabled(dropoutLayer, true);
    Tensor Z = dropoutLayer.forwardBatch(X);
    Scalar between = FrobeniusNorm.between(X, Z);
    assertTrue(Scalars.lessEquals(RealScalar.ONE, between));
  }

  @Test
  void testDistribution() {
    Distribution distribution = DropoutLayer.distribution(RealScalar.of(0.4));
    Tensor mean = Mean.of(RandomVariate.stream(distribution).limit(2000));
    Chop._01.requireClose(RealScalar.ONE, mean);
  }

  @Test
  void testDistribution2() {
    Distribution distribution = DropoutLayer.distribution(Rational.THIRD);
    Tensor tensor = Tensor.of(RandomVariate.stream(distribution).limit(100).distinct().sorted());
    assertEquals(tensor.toString(), "{0, 3/2}");
  }

  @Test
  void testNoDrop() {
    NetLayer dropoutLayer = DropoutLayer.of(0.0);
    Tensor X = RandomVariate.of(NormalDistribution.standard(), 3, 4, 5);
    Tensor Y = dropoutLayer.forwardBatch(X);
    Tolerance.CHOP.requireClose(X, Y);
    DropoutLayers.setEnabled(dropoutLayer, true);
    Tensor Z = dropoutLayer.forwardBatch(X);
    Tolerance.CHOP.requireClose(X, Z);
  }

  @Test
  void testBackWithNullMaskAfterDisableOrDeserialization() {
    DropoutLayer layer = (DropoutLayer) DropoutLayer.of(0.3);
    layer.setEnabled(true);
    DropoutLayer copy = Serialization.copy(layer);
    assertFalse(copy.isEnabled());
    Tensor grad = Tensors.vector(1.0, 2.0);
    assertEquals(copy.back(grad), grad);
  }

  @Test
  void testInvalidProbabilities() {
    assertThrows(Exception.class, () -> DropoutLayer.of(-0.1));
    assertThrows(Exception.class, () -> DropoutLayer.of(1.0));
    assertThrows(Exception.class, () -> DropoutLayer.of(1.5));
    assertThrows(Exception.class, () -> DropoutLayer.of((Number) null));
  }
}
