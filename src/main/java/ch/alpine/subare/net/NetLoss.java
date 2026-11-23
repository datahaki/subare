// code by jph
package ch.alpine.subare.net;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

public record NetLoss(Scalar loss, Tensor gradOutput) {
}
