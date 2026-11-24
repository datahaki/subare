package ch.alpine.subare.net.lay;

import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.net.reg.NoRegularization;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.ConstantArray;
import ch.alpine.tensor.mat.Tolerance;

class SwiGLULayerMathTest {
  @Test
  void testReplicatedBatchMatchesSingleSampleMath() {
    RandomGenerator rng = RandomGeneratorFactory.of("L64X256MixRandom").create(11L);
    int batchSize = 3;
    SwiGLULayer single = new SwiGLULayer(2, 4, 3);
    single.netRegularization = NoRegularization.INSTANCE;
    single.initialize(rng);
    SwiGLULayer batched = new SwiGLULayer(2, 4, 3);
    batched.netRegularization = NoRegularization.INSTANCE;
    batched.W_gate = single.W_gate.copy();
    batched.W_val = single.W_val.copy();
    batched.W_out = single.W_out.copy();
    Tensor x = Tensors.vector(-0.7, 1.2);
    Tensor gradOut = Tensors.vector(0.4, -0.9, 0.25);
    Tensor ySingle = single.forward(x);
    Tensor dxSingle = single.back(gradOut);
    Tensor gradsSingle = single.gradient(false);
    Tensor xBatch = ConstantArray.of(x, batchSize);
    Tensor gradOutBatch = ConstantArray.of(gradOut, batchSize);
    Tensor yBatch = batched.forwardBatch(xBatch);
    Tensor dxBatch = batched.backBatch(gradOutBatch);
    Tensor gradsBatch = batched.gradient(true);
    for (int b = 0; b < batchSize; ++b) {
      Tolerance.CHOP.requireClose(yBatch.get(b), ySingle);
      Tolerance.CHOP.requireClose(dxBatch.get(b), dxSingle);
    }
    for (int m = 0; m < 3; ++m) {
      Tensor scaledSingle = gradsSingle.get(m).multiply(RealScalar.of(batchSize));
      Tolerance.CHOP.requireClose(gradsBatch.get(m), scaledSingle);
    }
  }
}
