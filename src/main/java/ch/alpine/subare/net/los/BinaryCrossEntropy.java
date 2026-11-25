// code by gemini
package ch.alpine.subare.net.los;

import java.io.Serializable;
import java.util.List;

import ch.alpine.tensor.Rational;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.ext.Int;
import ch.alpine.tensor.ext.Integers;
import ch.alpine.tensor.io.MathematicaFormat;
import ch.alpine.tensor.red.Mean;
import ch.alpine.tensor.sca.exp.LogisticSigmoid;

/** implementation is specialized for Scalar output */
public enum BinaryCrossEntropy implements LossFunction, Serializable {
  SCALAR {
    /** @param logits raw logit Scalar
     * @param target scalar (0 or 1, or soft label in [0, 1])
     * @return loss and gradient */
    @Override
    public NetLoss compute(Tensor logits, Tensor target) {
      Scalar z = (Scalar) logits;
      Scalar y = (Scalar) target;
      return new NetLoss( //
          BceFromLogit.FUNCTION.apply(z, y), //
          LogisticSigmoid.FUNCTION.apply(z).subtract(y));
    }

    /** @param logits batch tensor of raw logits (vector of scalars or matrix for multi-label)
     * @param target vector of targets matching batch size
     * @return mean loss and batch-scaled gradient tensor with respect to logits _p */
    @Override
    public NetLoss computeBatch(Tensor logits, Tensor target) {
      int batch_size = Integers.requireEquals(logits.length(), target.length());
      Int myI = new Int();
      List<NetLoss> list = logits.stream() //
          .map(pred -> compute(pred, target.Get(myI.getAndIncrement()))) //
          .toList();
      Scalar loss = (Scalar) Mean.of(list.stream().map(NetLoss::loss));
      Scalar factor = Rational.of(1, batch_size);
      Tensor gradOutput = Tensor.of(list.stream() //
          .map(NetLoss::gradOutput) //
          .map(grad -> grad.multiply(factor)), batch_size);
      return new NetLoss(loss, gradOutput);
    }
  };

  @Override
  public final String toString() {
    return MathematicaFormat.concise("BinaryCrossEntropy", name());
  }
}
