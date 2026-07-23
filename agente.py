import numpy as np
import matplotlib.pyplot as plt


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

    def decidir(self, recompensa_por_explorar_seguro, castigo_por_explorar_peligro):
        """Decide si explorar o evitar comparando el valor esperado de cada opción.

        Regla de valor esperado (Opción B): el agente calcula cuánto le
        "conviene" en promedio explorar y lo compara con lo que le cuesta
        evitar. Elige la opción de mayor valor.

        La tolerancia a la incertidumbre entra como un "factor de pesimismo":
        cuanto MENOS tolerante es el agente, MÁS grande siente el peligro.
        """
        # Factor de pesimismo = piso de 1 (ver el peligro normal) + lo intolerante que es.
        factor_pesimismo = 2 - self.tolerancia_a_incertidumbre
        castigo_sentido = castigo_por_explorar_peligro * factor_pesimismo

        # Valor de explorar: promedio ponderado de sus dos finales (seguro vs. peligro).
        valor_explorar = (
            (1 - self.creencia_amenaza) * recompensa_por_explorar_seguro
            + self.creencia_amenaza * castigo_sentido
        )
        # Valor de evitar: siempre pierde su costo de evitación, pase lo que pase.
        valor_evitar = -self.costo_de_evitacion

        if valor_explorar >= valor_evitar:
            return "explorar"
        else:
            return "evitar"


def simular(agente, observaciones, recompensa_por_explorar_seguro, castigo_por_explorar_peligro):
    """Corre la simulación completa de UN agente sobre una secuencia de observaciones.

    Devuelve tres listas con el historial: la creencia, la decisión y el
    marcador acumulado en cada ensayo.
    """
    marcador = 0.0
    historial_creencia = []
    historial_decision = []
    historial_marcador = []

    for observacion in observaciones:
        # 1. El agente decide con lo que cree AHORA (antes de ver la observación).
        decision = agente.decidir(recompensa_por_explorar_seguro, castigo_por_explorar_peligro)
        # 2. Aprende de lo que el mundo le mostró y ajusta su termómetro.
        creencia = agente.actualizar_creencia(observacion)

        # 3. Reparte los puntos de esta asomada según lo que hizo y lo que había.
        if decision == "evitar":
            puntos = -agente.costo_de_evitacion
        elif observacion == 0:
            puntos = recompensa_por_explorar_seguro
        else:
            puntos = castigo_por_explorar_peligro

        marcador = marcador + puntos

        # 4. Anota el historial de este ensayo.
        historial_creencia.append(creencia)
        historial_decision.append(decision)
        historial_marcador.append(marcador)

    return historial_creencia, historial_decision, historial_marcador


# ---------------------------------------------------------------------------
# Simulación de la Fase 5a: la misma corrida, ahora a través de la función simular().
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

    # Generamos UNA secuencia de observaciones que enfrentarán los agentes,
    # para que el mundo sea idéntico y la comparación sea justa.
    observaciones = []
    for _ in range(numero_de_ensayos):
        observaciones.append(1 if generador.random() < peligrosidad_real else 0)

    agente_sano = Agente(
        creencia_inicial_amenaza=0.5,      # Prior neutral: "no sé, 50/50".
        tasa_aprendizaje_amenaza=0.3,      # Aprende el peligro a velocidad moderada.
        tasa_aprendizaje_seguridad=0.3,    # Aprende la seguridad a la MISMA velocidad (agente equilibrado).
        tolerancia_a_incertidumbre=0.5,    # Aguanta hasta 50% de miedo antes de evitar (valiente y equilibrado).
        costo_de_evitacion=0.5,            # Precio de jugar a lo seguro cada vez que evita.
    )

    # Corremos la simulación del agente SANO y recogemos sus tres historiales.
    historial_creencia_sano, historial_decision_sano, historial_marcador_sano = simular(
        agente_sano,
        observaciones,
        recompensa_por_explorar_seguro,
        castigo_por_explorar_peligro,
    )

    # El agente ANSIOSO: la MISMA clase Agente, solo cambian los parámetros.
    agente_ansioso = Agente(
        creencia_inicial_amenaza=0.7,      # Arranca temeroso (prior de peligro alto).
        tasa_aprendizaje_amenaza=0.4,      # Aprende el peligro rápido.
        tasa_aprendizaje_seguridad=0.15,   # Aprende la seguridad lento: el miedo tarda en bajar.
        tolerancia_a_incertidumbre=0.25,   # Poco tolerante: siente el peligro más grande (pesimismo 1.75).
        costo_de_evitacion=0.5,            # MISMO costo que el sano (para una comparación justa).
    )

    # Corremos al ANSIOSO sobre EL MISMO mundo (las mismas observaciones).
    historial_creencia_ansioso, historial_decision_ansioso, historial_marcador_ansioso = simular(
        agente_ansioso,
        observaciones,
        recompensa_por_explorar_seguro,
        castigo_por_explorar_peligro,
    )

    # Comparación numérica: marcador final y cuántas veces evitó cada uno.
    marcador_final_sano = historial_marcador_sano[-1]
    marcador_final_ansioso = historial_marcador_ansioso[-1]
    evitaciones_sano = historial_decision_sano.count("evitar")
    evitaciones_ansioso = historial_decision_ansioso.count("evitar")

    print(f"Agente SANO    -> marcador: {marcador_final_sano:+.1f} | evitó {evitaciones_sano} de {numero_de_ensayos} veces")
    print(f"Agente ANSIOSO -> marcador: {marcador_final_ansioso:+.1f} | evitó {evitaciones_ansioso} de {numero_de_ensayos} veces")

    # -----------------------------------------------------------------------
    # Gráfica comparativa: los dos agentes en los mismos ejes.
    # -----------------------------------------------------------------------
    ensayos = range(1, numero_de_ensayos + 1)   # eje horizontal: 1, 2, ..., 100

    # Colores validados como seguros para daltonismo (azul sano / morado ansioso).
    color_sano = "#0072B2"
    color_ansioso = "#762A83"

    figura, (panel_miedo, panel_marcador) = plt.subplots(2, 1, figsize=(10, 8))

    # Panel de arriba: el termómetro del miedo de AMBOS agentes.
    panel_miedo.plot(ensayos, historial_creencia_sano, color=color_sano, label="Sano")
    panel_miedo.plot(ensayos, historial_creencia_ansioso, color=color_ansioso, label="Ansioso")
    panel_miedo.axhline(peligrosidad_real, color="gray", linestyle="--", label="Peligrosidad real (0.2)")
    panel_miedo.set_title("Cómo evoluciona el miedo: sano vs. ansioso")
    panel_miedo.set_ylabel("Creencia de amenaza (0 a 1)")
    panel_miedo.set_ylim(0, 1)
    panel_miedo.legend()
    panel_miedo.grid(True, alpha=0.3)

    # Panel de abajo: el marcador acumulado de AMBOS agentes.
    panel_marcador.plot(ensayos, historial_marcador_sano, color=color_sano, label="Sano")
    panel_marcador.plot(ensayos, historial_marcador_ansioso, color=color_ansioso, label="Ansioso")
    panel_marcador.axhline(0, color="gray", linestyle="--")   # línea de referencia en cero
    panel_marcador.set_title("Marcador acumulado: sano vs. ansioso")
    panel_marcador.set_xlabel("Ensayo")
    panel_marcador.set_ylabel("Puntaje acumulado")
    panel_marcador.legend()
    panel_marcador.grid(True, alpha=0.3)

    figura.tight_layout()                 # acomoda los paneles para que no se encimen
    figura.savefig("comparacion.png", dpi=120)
    print("Grafica comparativa guardada en: comparacion.png")
