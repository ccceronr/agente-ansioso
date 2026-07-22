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
        tolerancia_a_incertidumbre,
        costo_de_evitacion,
    ):
        # Estado que EVOLUCIONA con la experiencia (arranca en el valor inicial).
        self.creencia_amenaza = creencia_inicial_amenaza

        # Parámetros FIJOS que definen la "personalidad" del agente.
        self.tasa_aprendizaje_amenaza = tasa_aprendizaje_amenaza
        self.tasa_aprendizaje_seguridad = tasa_aprendizaje_seguridad
        self.tolerancia_a_incertidumbre = tolerancia_a_incertidumbre
        self.costo_de_evitacion = costo_de_evitacion

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

    def decidir(self):
        """Decide si explorar o evitar según el nivel de miedo actual.

        Regla de umbral (Opción A): si el miedo (creencia_amenaza) supera lo
        que el agente está dispuesto a tolerar (tolerancia_a_incertidumbre),
        se aleja (evita). Si no, se acerca (explora).
        """
        if self.creencia_amenaza > self.tolerancia_a_incertidumbre:
            return "evitar"
        else:
            return "explorar"


# ---------------------------------------------------------------------------
# Simulación de prueba de la Fase 2: el agente sano aprende Y decide.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Generador de números aleatorios con "semilla" fija: garantiza que la
    # simulación dé el mismo resultado cada vez que se corre (reproducibilidad).
    generador = np.random.default_rng(42)

    peligrosidad_real = 0.2      # El mundo es peligroso el 20% de las veces (el agente NO lo sabe).
    numero_de_ensayos = 100

    # Reglas de puntos DEL MUNDO (cuánto premia o castiga el entorno).
    recompensa_por_explorar_seguro = 1.0    # Exploró y estaba seguro: premio.
    castigo_por_explorar_peligro = -1.0     # Exploró y había peligro: susto.

    agente_sano = Agente(
        creencia_inicial_amenaza=0.5,      # Prior neutral: "no sé, 50/50".
        tasa_aprendizaje_amenaza=0.3,      # Aprende el peligro a velocidad moderada.
        tasa_aprendizaje_seguridad=0.3,    # Aprende la seguridad a la MISMA velocidad (agente equilibrado).
        tolerancia_a_incertidumbre=0.5,    # Aguanta hasta 50% de miedo antes de evitar (valiente y equilibrado).
        costo_de_evitacion=0.5,            # Precio de jugar a lo seguro cada vez que evita.
    )

    marcador = 0.0    # El "puntaje" acumulado del robot; empieza en cero.

    for ensayo in range(numero_de_ensayos):
        # 1. El robot decide con lo que cree AHORA (antes de ver qué había).
        decision = agente_sano.decidir()
        # 2. El mundo revela lo que había: peligro (1) con probabilidad 0.2.
        observacion = 1 if generador.random() < peligrosidad_real else 0
        # 3. El robot aprende de lo que vio y ajusta su termómetro.
        creencia = agente_sano.actualizar_creencia(observacion)

        # 4. Repartimos los puntos de esta asomada según lo que hizo y lo que había.
        if decision == "evitar":
            puntos = -agente_sano.costo_de_evitacion          # Evitó: paga su precio, pase lo que pase.
        elif observacion == 0:
            puntos = recompensa_por_explorar_seguro           # Exploró y estaba seguro: premio.
        else:
            puntos = castigo_por_explorar_peligro             # Exploró y había peligro: susto.

        marcador = marcador + puntos

        print(
            f"Ensayo {ensayo + 1:3d} | decision: {decision:8s} | "
            f"observacion: {observacion} | creencia: {creencia:.3f} | "
            f"puntos: {puntos:+.1f} | marcador: {marcador:+.1f}"
        )

    print(f"\nMarcador final del agente sano: {marcador:+.1f}")
