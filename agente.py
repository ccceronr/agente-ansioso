import numpy as np


class Agente:
    """Agente que aprende a estimar qué tan peligrosa es una situación.

    En cada ensayo recibe una observación (peligro=1 o seguro=0) y ajusta
    su creencia de amenaza usando la regla de error de predicción
    (Rescorla-Wagner), con tasas de aprendizaje distintas para el peligro
    y para la seguridad.
    """

    def __init__(
        self,
        creencia_inicial_amenaza,
        tasa_aprendizaje_amenaza,
        tasa_aprendizaje_seguridad,
    ):
        # Estado que EVOLUCIONA con la experiencia (arranca en el valor inicial).
        self.creencia_amenaza = creencia_inicial_amenaza

        # Parámetros FIJOS que definen la "personalidad" del agente.
        self.tasa_aprendizaje_amenaza = tasa_aprendizaje_amenaza
        self.tasa_aprendizaje_seguridad = tasa_aprendizaje_seguridad

    def actualizar_creencia(self, observacion):
        """Ajusta la creencia de amenaza a partir de una observación."""
        # La "sorpresa": qué tan lejos estuvo la realidad de lo que esperaba.
        error_prediccion = observacion - self.creencia_amenaza

        # Aprendizaje asimétrico: usamos una tasa distinta según la dirección
        # de la sorpresa (hacia el peligro vs. hacia la seguridad).
        if error_prediccion >= 0:
            tasa = self.tasa_aprendizaje_amenaza     # el mundo fue MÁS peligroso de lo esperado
        else:
            tasa = self.tasa_aprendizaje_seguridad   # el mundo fue MÁS seguro de lo esperado

        # Regla de Rescorla-Wagner: corrijo mi creencia en proporción a la sorpresa.
        self.creencia_amenaza = self.creencia_amenaza + tasa * error_prediccion

        return self.creencia_amenaza


# ---------------------------------------------------------------------------
# Simulación de prueba de la Fase 1: ver al agente sano aprender.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Generador de números aleatorios con "semilla" fija: garantiza que la
    # simulación dé el mismo resultado cada vez que se corre (reproducibilidad).
    generador = np.random.default_rng(42)

    peligrosidad_real = 0.2      # El mundo es peligroso el 20% de las veces (el agente NO lo sabe).
    numero_de_ensayos = 100

    agente_sano = Agente(
        creencia_inicial_amenaza=0.5,      # Prior neutral: "no sé, 50/50".
        tasa_aprendizaje_amenaza=0.3,      # Aprende el peligro a velocidad moderada.
        tasa_aprendizaje_seguridad=0.3,    # Aprende la seguridad a la MISMA velocidad (agente equilibrado).
    )

    for ensayo in range(numero_de_ensayos):
        # El mundo lanza una "moneda cargada": peligro (1) con probabilidad 0.2.
        observacion = 1 if generador.random() < peligrosidad_real else 0
        creencia = agente_sano.actualizar_creencia(observacion)
        print(f"Ensayo {ensayo + 1:3d} | observacion: {observacion} | creencia de amenaza: {creencia:.3f}")
