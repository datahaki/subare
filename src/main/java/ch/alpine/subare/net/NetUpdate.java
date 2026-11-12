package ch.alpine.subare.net;

import ch.alpine.tensor.Scalar;

public record NetUpdate(NetLayer netLayer, Scalar lr) {
  public void with() {
    netLayer.stream().forEach(this::check);
  }

  private void check(NetLayer netLayer) {
    if (netLayer instanceof LinearLayer ll) {
      ll.W = ll.W.add(ll.gW.multiply(lr.negate()));
      ll.b = ll.b.add(ll.gb.multiply(lr.negate()));
    }
  }
}
