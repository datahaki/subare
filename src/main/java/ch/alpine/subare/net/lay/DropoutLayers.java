// code by jph
package ch.alpine.subare.net.lay;

public enum DropoutLayers {
  ;
  public static void setEnabled(NetLayer netLayer, boolean enabled) {
    NetExtract.flatten(netLayer) //
        .filter(DropoutLayer.class::isInstance) //
        .map(DropoutLayer.class::cast) //
        .forEach(dropoutLayer -> dropoutLayer.setEnabled(enabled));
  }
}
