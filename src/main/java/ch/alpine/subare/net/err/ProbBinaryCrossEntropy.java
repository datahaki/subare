package ch.alpine.subare.net.err;

import java.io.Serializable;
import java.util.List;

import ch.alpine.subare.net.NetLoss;
import ch.alpine.tensor.Rational;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.ext.Int;
import ch.alpine.tensor.ext.Integers;
import ch.alpine.tensor.io.MathematicaFormat;
import ch.alpine.tensor.red.Mean;
import ch.alpine.tensor.sca.Clip;
import ch.alpine.tensor.sca.Clips;

/** Standalone probability BCE helper taking (logit, activation, target) without implementing LossFunction */
public enum ProbBinaryCrossEntropy implements Serializable {
  SCALAR;

  private static final Clip CLIP = Clips.interval(1E-15, 1 - 1E-15);

  /** @param logit raw logit Scalar
   * @param activation predicted probability Scalar
   * @param target scalar (0 or 1, or soft label in [0, 1])
   * @return loss (from logit) and gradient dL/dp (from clipped activation) */
  public static NetLoss compute(Scalar z, Scalar p, Scalar y) {
    p = CLIP.apply(p);
    Scalar oneMinusP = RealScalar.ONE.subtract(p);
    Scalar oneMinusY = RealScalar.ONE.subtract(y);
    Scalar loss = BceFromLogit.FUNCTION.apply(z, y);
    Scalar gradP = y.divide(p).negate().add(oneMinusY.divide(oneMinusP));
    return new NetLoss(loss, gradP);
  }

  /** @param logit batch tensor of raw logits
   * @param activation batch tensor of predicted probabilities
   * @param target vector of targets matching batch size
   * @return mean loss and batch-scaled gradient tensor with respect to activation probabilities */
  public static NetLoss computeBatch(Tensor logit, Tensor activation, Tensor target) {
    int batch_size = Integers.requireEquals(activation.length(), target.length());
    Integers.requireEquals(logit.length(), batch_size);
    Int myI = new Int();
    List<NetLoss> list = activation.stream() //
        .map(act -> {
          int idx = myI.getAndIncrement();
          return compute(logit.Get(idx), (Scalar) act, target.Get(idx));
        }).toList();
    Scalar loss = (Scalar) Mean.of(list.stream().map(NetLoss::loss));
    Scalar factor = Rational.of(1, batch_size);
    Tensor gradOutput = Tensor.of(list.stream() //
        .map(NetLoss::gradOutput) //
        .map(grad -> grad.multiply(factor)), batch_size);
    return new NetLoss(loss, gradOutput);
  }

  @Override
  public final String toString() {
    return MathematicaFormat.concise("ProbBinaryCrossEntropy", name());
  }
}
