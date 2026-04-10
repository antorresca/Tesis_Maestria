PRESENTACIÓN PROPUESTA
7.	Título: Integración de Aprendizaje por Refuerzo Profundo y Lenguaje Natural en Robots Móviles Manipuladores
8.	Línea de investigación: Robots y Sistemas Autónomos.
9.	Introducción:
Los robots manipuladores móviles representan una tecnología crítica en la robótica de servicio e industrial, donde la capacidad de operar en entornos dinámicos y colaborar con seres humanos es fundamental. Para alcanzar un nivel de autonomía operativa superior, estos sistemas deben trascender las secuencias de control preprogramadas y ser capaces de interpretar instrucciones humanas de lenguaje natural no estructurado (comandos de alto nivel semántico como "lleva el objeto a la mesa") para planificar y ejecutar tareas complejas.
El Aprendizaje por Refuerzo Profundo (DRL) ha emergido como una metodología robusta para la generación de políticas de control en navegación y manipulación [1], [2], [3]. No obstante, la implementación práctica de estas políticas enfrenta desafíos significativos en cuanto a la eficiencia de muestreo y la interpretación de la intención humana. La mayoría de los enfoques convencionales requieren objetivos definidos en espacios de estados numéricos, careciendo de mecanismos para traducir el lenguaje natural en acciones físicas ejecutables [1], [4].
Por otro lado, el Procesamiento de Lenguaje Natural (PLN) permite realizar una interpretación semántica de las instrucciones [5], [6], [7]. La integración de ambas áreas, mediante arquitecturas jerárquicas o desacopladas, permite separar el razonamiento simbólico de alto nivel de la ejecución motora de bajo nivel [8], [9]. 
En este contexto, se propone el desarrollo de una arquitectura de planificación de tareas que articule módulos de PLN y DRL. El sistema se fundamenta en un modelo robótico de manipulación móvil basado en la plataforma Robotino y actuadores Dynamixel [10] , el cual es migrado a un entorno de simulación física de alto desempeño (como PyBullet o similares) para facilitar el entrenamiento masivo de agentes. La arquitectura propuesta contempla un enfoque modular donde la interpretación de comandos se vincula con políticas de control especializadas para movilidad y manipulación. Esta estructura permite la escalabilidad hacia esquemas de Aprendizaje por Refuerzo Jerárquico (HRL) y la integración de mecanismos de anclaje sensorial para el reconocimiento del estado del entorno [11]. El desempeño del sistema será evaluado mediante métricas de tasa de éxito, eficiencia de trayectoria y adaptabilidad ante cambios en la configuración del escenario, estableciendo un marco de trabajo para robots capaces de cerrar el ciclo entre la instrucción lingüística y la acción autónoma.
10.	Antecedentes: 
•	Aprendizaje por Refuerzo Profundo en robótica móvil manipuladora
El Aprendizaje por Refuerzo Profundo (DRL) se ha consolidado como una herramienta fundamental para el desarrollo de políticas de control en sistemas robóticos de manipulación móvil. Es especialmente relevante en escenarios caracterizados por espacios de estado de alta dimensionalidad donde los modelos analíticos resultan limitados. Recientemente, se ha demostrado que el éxito del DRL depende de la capacidad de aprender representaciones de estado que mejoren la eficiencia de muestreo y la generalización en entornos complejos [12].
Uno de los principales desafíos es la eficiencia de muestreo y el manejo de recompensas dispersas. Diversos trabajos han demostrado que técnicas como la reutilización de experiencias y el aprendizaje condicionado por metas (Goal-Conditioned RL) permiten aprender políticas robustas reduciendo las interacciones necesarias [1], [4] Investigaciones recientes sugieren que el uso de currículos de metas autogenerados y el pre-entrenamiento con demostraciones facilitan la convergencia en tareas de múltiples pasos [13]. Por otra parte, estudios recientes refuerzan el rol del DRL como mecanismo de ejecución motora y adaptación local, señalando que su evolución depende de una integración en arquitecturas de planificación más estructuradas [14], [15].
En la arquitectura propuesta, se aborda esta limitación integrando el módulo de PLN para proveer objetivos estructurados, permitiendo que el agente se enfoque en regiones del espacio de estados relevantes para la instrucción humana.
•	Arquitecturas integradas para planificación de tareas
Las arquitecturas integradas estructuran la relación entre intención, planificación y ejecución mediante esquemas jerárquicos. En este ámbito, se destacan estos enfoques clave:
o	Descomposición de Tareas: El aprendizaje alcanza mayores tasas de éxito al dividir tareas complejas en sub-tareas con sistemas de recompensa específicos, superando los enfoques tradicionales end-to-end [16].
o	Planificación Semántica con LLMs: Los modelos de lenguaje actúan como intermediarios para la descomposición de tareas y la generación de objetivos intermedios sin asumir el control motor directo [8], [17].
o	Aprendizaje desde el Éxito y el Fallo: La incorporación de mecanismos de retroalimentación negativa y el reetiquetado de metas permite que el sistema mantenga la robustez frente a cambios imprevistos [18].
o	Modularidad y Aprendizaje Continuo: El aprendizaje en entornos reales alcanza mayores tasas de éxito mediante arquitecturas que encapsulan mecanismos de control aprendidos dentro de estructuras de planificación más amplias, utilizando recompensas genéricas basadas en percepción [19].
La arquitectura desarrollada adopta estos principios mediante un diseño modular donde el módulo de PLN genera la descomposición semántica, mientras que agentes especializados de DRL ejecutan acciones motoras específicas bajo un esquema de control desacoplado.
•	Procesamiento de Lenguaje Natural para interacción humano-robot
El procesamiento de lenguaje natural (PLN) permite transformar instrucciones de alto nivel en representaciones semánticas accionables. La literatura identifica tres vertientes principales:
o	Razonamiento con LLMs: Integración de modelos de gran escala para la generación de planes abstractos y refinamiento de tareas en tiempo real [20].
o	Arquitecturas Contextuales: Sistemas que integran lenguaje y percepción para el razonamiento espacial, gestionando la ambigüedad en la toma de decisiones [5].
o	Lenguaje como Recompensa: Uso de descripciones lingüísticas como retroalimentación estructurada para mejorar la interpretabilidad [6], [7].
Para la arquitectura propuesta, el módulo de PLN actúa como un traductor que convierte instrucciones en un lenguaje intermedio estructurado. Esta formalización permite que el sistema opere mediante Goal-Conditioned RL, donde las metas se definen semánticamente y se ejecutan mediante políticas de acción discretas (DQN) para garantizar estabilidad y eficiencia computacional.
•	Sistemas multimodales en robótica de servicio
Los sistemas multimodales integran visión, lenguaje y acción para operar en entornos no estructurados. Se destacan:
o	Modelos de Visión-Lenguaje-Acción (VLA): Arquitecturas como RT-2 [21] que, aunque potentes en razonamiento, presentan limitaciones en la precisión de la interacción física fina [22], [23].
o	Representaciones basadas en Affordances: Modelos centrados en las posibilidades de interacción de los objetos para fortalecer la ejecución motora y la generalización [24].
o	Aprendizaje por Refuerzo Multimodal: La integración de señales de voz, percepción visual y acciones en bucles cerrados ha demostrado ventajas críticas para manejar la incertidumbre en contextos de asistencia y colaboración humano-robot [25].
o	Representaciones de Estado Eficientes: El uso de técnicas de aprendizaje de representaciones para extraer características críticas de sensores heterogéneos, facilitando la toma de decisiones en horizontes temporales prolongados [11], [12].
11.	Planteamiento del problema: 
Los robots manipuladores móviles en entornos de servicio requieren operar bajo instrucciones humanas de alto nivel para superar la rigidez de las interfaces tradicionales. No obstante, persiste una brecha crítica: la dependencia de secuencias predefinidas limita la autonomía en escenarios no estructurados. El desafío técnico no es solo la comprensión del mensaje o el movimiento del hardware de forma aislada, sino la ausencia de arquitecturas integradas que articulen la interpretación semántica y la ejecución motora autónoma en entornos donde la predictibilidad es limitada.
La problemática reside en la fragmentación de las soluciones actuales. Por un lado, el Aprendizaje por Refuerzo Profundo (DRL) es eficaz para el control motor, pero enfrenta desafíos en la eficiencia de muestreo y la gestión de representaciones de estado complejas [12]. Por otro lado, el Procesamiento de Lenguaje Natural (PLN) ofrece interpretación semántica, pero a menudo carece de un anclaje (grounding) físico, generando planes que ignoran las restricciones cinemáticas o dinámicas del robot [5]. . Si bien los modelos de Visión-Lenguaje-Acción (VLA) de gran escala intentan cerrar esta brecha, presentan dificultades en la precisión física requerida para la manipulación fina y una alta demanda computacional [21], [23].
Esta desconexión impide que el sistema gestione la ambigüedad lingüística o se adapte a tareas de largo horizonte. Sin una estructura que vincule la semántica con la dinámica del entorno, el robot no puede cerrar el bucle de control ante cambios imprevistos. Investigaciones recientes sugieren que la descomposición de tareas en sub-objetivos específicos es una estrategia superior para alcanzar el éxito en tareas de alto nivel comparado con enfoques globales [16].
Para resolver esta brecha, se propone una arquitectura de planificación modular y jerárquica. En este esquema, modelos de lenguaje preentrenados basados en arquitectura Transformer que traducirá las instrucciones en representaciones estructuradas de metas. Estas metas serán procesadas por un módulo de generación de metas para asegurar el anclaje físico, permitiendo que el agente de control opere bajo un enfoque de aprendizaje condicionado por metas (Goal-Conditioned RL), técnica que facilita el aprendizaje desde el éxito y el fallo en entornos con recompensas dispersas [18].
La implementación técnica tomará como base el modelo robótico institucional descrito en [10], el cual será migrado hacia un entorno de simulación física de alto desempeño para facilitar el entrenamiento masivo. Se evaluará el desempeño mediante la implementación de al menos dos técnicas de DRL representativas (como DQN y PPO), determinando la política más robusta ante la variabilidad del entorno y las restricciones de hardware. La validez de esta solución se establecerá midiendo la capacidad de ejecución autónoma y la efectividad del sistema mediante métricas de tasa de éxito, eficiencia de trayectoria y precisión [11].
12.	Justificación: 
La ejecución de este trabajo se justifica por la necesidad técnica de dotar a los sistemas de manipulación móvil de una capa de aplicación que trascienda la programación de secuencias rígidas. En el ámbito de la ingeniería aplicada, el diseño de arquitecturas adaptables es fundamental para el despliegue de robots en entornos dinámicos, donde la predictibilidad es limitada y la interacción humana es constante.
Desde la perspectiva del desarrollo tecnológico y académico, este proyecto implementa un esquema jerárquico que permite un desacoplamiento funcional: mientras el módulo de DRL garantiza la destreza física, el módulo de PLN gestiona la lógica de la tarea [3]. La novedad de esta propuesta radica en el uso de modelos lingüísticos basados en arquitectura Transformer para la generación de metas estructuradas (Goal-Conditioned RL). Según la literatura reciente, el uso de representaciones de estado eficientes y la guía lingüística pueden derivar en una mejora significativa de la eficiencia de muestreo y la capacidad de generalización en comparación con métodos que no utilizan guía semántica [6], [12]. Este enfoque contribuye directamente a la discusión sobre modelos de Visión-Lenguaje-Acción (VLA), proporcionando una alternativa modular a los modelos end-to-end masivos. Al descomponer la tarea en sub-objetivos específicos, se favorece la interpretabilidad y se reduce la carga computacional [16], [21]
En el contexto práctico, la relevancia de esta solución es significativa para el sector de servicios y la industria 4.0 en Colombia. El desarrollo de sistemas autónomos capaces de interpretar instrucciones naturales reduce la barrera técnica para operarios no especializados, facilitando la adopción de tecnologías robóticas en logística hospitalaria o asistencia en almacenes. Para asegurar la viabilidad técnica, se utiliza como base el modelo robótico desarrollado en [10] , el cual es migrado hacia entornos de simulación de alto desempeño y eficiencia. Esta transición garantiza un marco de desarrollo de bajo costo, alta seguridad y total reproducibilidad, permitiendo la maduración de la arquitectura y el aprendizaje tanto de los éxitos como de los fallos del sistema antes de una futura implementación en hardware real [18].
Finalmente, la importancia de este trabajo se cuantificará mediante la evaluación de métricas de desempeño técnico alineadas con los estándares de la industria: la tasa de éxito en la manipulación, los tiempos de ejecución, la eficiencia de trayectoria y la adaptabilidad ante cambios imprevistos en el escenario (Fase 5). De esta manera, el proyecto no solo resuelve un problema de integración modular, sino que entrega una solución funcional que posiciona el uso de tecnologías de vanguardia como una herramienta viable para mejorar la competitividad en entornos dinámicos [11].
13.	Objetivo general y objetivos específicos:
Objetivo General
Desarrollar una arquitectura de planeación de tareas para un sistema robótico de manipulación móvil que integre Aprendizaje por Refuerzo Profundo y Procesamiento de Lenguaje Natural, permitiendo la interpretación de instrucciones humanas y la ejecución autónoma de tareas de manipulación móvil mediante un enfoque de aprendizaje condicionado por metas en entornos simulados dinámicos.
Objetivos Específicos
1.	Implementar un módulo de aprendizaje por refuerzo profundo mediante la descomposición de tareas en sub-objetivos operativos utilizando al menos dos técnicas representativas aplicadas a tareas de manipulación y navegación móvil.
2.	Implementar un módulo de procesamiento de lenguaje natural, mediante modelos lingüísticos preentrenados, orientado a la clasificación de intenciones y extracción de entidades semántica de instrucciones humanas para tareas de manipulación móvil.
3.	Integrar los módulos de aprendizaje por refuerzo profundo y de procesamiento de lenguaje natural en la arquitectura de planificación de tareas, permitiendo la conversión de instrucciones humanas en secuencias de acciones de manipulación móvil.
4.	Evaluar el desempeño del sistema mediante las métricas de tasa de éxito, eficiencia de trayectoria, robustez ante colisiones y adaptabilidad ante cambios en el escenario y en las tareas de manipulación móvil realizadas en entornos simulados controlados.
14.	Metodología:
El presente proyecto adopta una metodología de diseño, desarrollo e implementación experimental de carácter aplicado, con un enfoque cuantitativo, orientada a la creación de una arquitectura de planificación de tareas para un sistema robótico de manipulación móvil. Esta arquitectura integra módulos de Procesamiento de Lenguaje Natural (PLN), representación de estados y Aprendizaje por Refuerzo Profundo (DRL) bajo un esquema de aprendizaje condicionado por metas (Goal-Conditioned RL).
El enfoque cuantitativo permite evaluar de manera objetiva, sistemática y reproducible el desempeño del sistema mediante métricas medibles, como la tasa de éxito en la ejecución de tareas, tiempo de ejecución, precisión en la manipulación y capacidad de adaptación a cambios en el entorno y las instrucciones. Estas métricas facilitan la comparación entre diferentes configuraciones del sistema y algoritmos implementados, así como el análisis de la interacción entre los módulos de razonamiento semántico y ejecución motora.
El carácter aplicado y experimental del proyecto busca desarrollar una solución funcional a un problema específico en robótica autónoma. La validación se llevará a cabo mediante la migración y puesta a punto de una plataforma robótica preexistente [10] hacia un entorno de simulación dinámico de alto desempeño, garantizando seguridad, repetibilidad y flexibilidad en la evaluación del sistema frente a escenarios no estructurados.
La metodología se organiza en seis fases secuenciales, que abarcan desde el diseño conceptual hasta la documentación final, asegurando la trazabilidad entre los objetivos planteados, las actividades realizadas y los resultados obtenidos.

15.	Actividades a desarrollar: 
a.	Fase 1. Diseño conceptual y análisis de requerimientos
Objetivos asociados: Todos
Entregables: Documento de requerimientos funcionales y técnicos, escenarios de simulación, criterios y métricas de evaluación.
Actividades principales:
-	A1. Identificación y definición de tareas de manipulación móvil.
-	 A2. Definición de tipos de instrucciones humanas (texto).
-	A3. Migración y configuración del modelo robótico institucional [10] (Robotino + Dynamixel) al nuevo entorno de simulación.
-	A4. Selección y configuración del entorno de simulación de alto desempeño.
-	A5. Definición de métricas cuantitativas de desempeño.
-	A6. Diseño de la arquitectura multimodal (PLN y DRL).
b.	Fase 2. Implementación del módulo de Procesamiento de Lenguaje Natural (PLN)
Objetivo asociado: Objetivo específico 2
Entregables: Módulo PLN funcional, dataset anotado, reporte de desempeño
Actividades principales:
-	A7. Selección e implementación del modelo basado en Transformers para clasificación de intenciones.
-	A8. Implementación del sistema de extracción de entidades para la definición de metas espaciales.
-	A9. Procesamiento y normalización de instrucciones mediante técnicas de PLN.
-	A10. Definición de la representación semántica y mapeo de objetivos espaciales.
-	A11. Evaluación del Módulo PLN por métricas de precisión.
c.	Fase 3. Implementación del módulo de Aprendizaje por Refuerzo Profundo (DRL)
Objetivo asociado: Objetivo específico 1
Entregables: Agentes DRL entrenados, scripts de entrenamiento, curvas de aprendizaje
Actividades principales:
-	A12. Selección de algoritmos DRL discretos y continuos (PPO, SAC, DQN).
-	A13. Diseño del entorno de entrenamiento (espacio de estados, acciones y funciones de recompensa).
-	A14. Entrenamiento y ajuste de hiperparámetros de los agentes.
-	A15. Análisis preliminar de la eficiencia de muestreo, convergencia y early stopping.
d.	Fase 4. Integración de los módulos PLN–DRL
Objetivo asociado: Objetivo específico 3
Entregables: Arquitectura integrada, registro de ejecuciones, informe de integración
Actividades principales:
-	A16. Definición de la interfaz o módulo de comunicación entre módulos.
-	A17. Traducción de representaciones semánticas en metas de control para el agente DRL.
-	A18. Sincronización del bucle de percepción con la política de ejecución motora.
-	A19. Validación funcional de la secuencia completa: Instrucción → Interpretación → Ejecución de sub-tareas.
e.	Fase 5. Validación experimental y evaluación
Objetivo asociado: Objetivo específico 4
Entregables: Informe de resultados, análisis comparativo, versión optimizada del sistema
Actividades principales:
-	A20. Diseño de escenarios de prueba dinámicos con obstáculos y metas aleatorias.
-	A21. Ejecución de experimentos y recolección de datos.
-	A22. Análisis estadístico de resultados (tasa de éxito, robustez ante colisiones y eficiencia de trayectoria).
-	A23. Identificación de limitaciones y comparación con modelos base.
f.	Fase 6. Documentación y difusión
Objetivos asociados: Todos
Entregables: Documento final de tesis, repositorio de código documentado
Actividades principales:
-	A24. Documentación técnica continua del desarrollo técnico.
-	A25. Gestión y limpieza del repositorio de código.
-	A26. Redacción y revisión del documento final de tesis.
16.	Cronograma: 
El desarrollo del proyecto se organiza en seis fases secuenciales y parcialmente paralelas, considerando las interdependencias entre ellas para garantizar un avance eficiente y coherente:
-	Fase 1. Diseño conceptual y análisis de requerimientos: Constituye la base técnica del proyecto. Incluye la definición de tareas y la migración de la plataforma robótica  [10] al nuevo entorno de simulación, proporcionando los elementos necesarios para la implementación de los módulos subsiguientes.
-	Fases 2 y 3. Implementación de los módulos de PLN y DRL: Estas fases dependen de los resultados de la Fase 1 y se desarrollarán en paralelo. Mientras se entrena el modelo de lenguaje para la clasificación de intenciones y extracción de parámetros (PLN), se procede simultáneamente con el diseño y entrenamiento de las políticas en el simulador (DRL). Este paralelismo permite avanzar en la comprensión semántica y la destreza física de forma independiente.
-	Fase 4. Integración de la Arquitectura Multimodal: Requiere la finalización de las Fases 2 y 3, ya que se centra en la sincronización funcional del flujo Lenguaje-Accion y la validación de la interfaz o módulo de comunicación entre módulos.
-	Fase 5. Validación experimental y evaluación: Depende de la arquitectura integrada de la Fase 4. Se orienta a la evaluación cuantitativa (tasa de éxito, colisiones, eficiencia) y al análisis del desempeño del sistema completo en escenarios dinámicos no estructurados.
-	Fase 6. Documentación y difusión: Se desarrolla de manera transversal a lo largo de todo el proyecto, asegurando la trazabilidad de los resultados, la gestión del repositorio de código y la consolidación del documento final de tesis.
Esta estructura permite visualizar claramente las dependencias y el paralelismo entre las fases, facilitando la planificación temporal y la asignación eficiente de recursos durante el desarrollo del proyecto.

19.	Bibliografía: 
[1]	Y. Zhou, Q. Feng, Y. Zhou, J. Lin, Z. Liu, and H. Wang, “Sample-Efficient Deep Reinforcement Learning of Mobile Manipulation for 6-DOF Trajectory Following,” IEEE Transactions on Automation Science and Engineering, vol. 22, pp. 11381–11391, 2025, doi: 10.1109/TASE.2025.3530162.
[2]	G. Kang, H. Seong, D. Lee, and D. H. Shim, “A versatile door opening system with mobile manipulator through adaptive position-force control and reinforcement learning,” Rob. Auton. Syst., vol. 180, Oct. 2024, doi: 10.1016/j.robot.2024.104760.
[3]	Y. Zhou, Y. Zhou, K. Jin, and H. Wang, “Hierarchical Reinforcement Learning With Model Guidance for Mobile Manipulation,” IEEE/ASME Transactions on Mechatronics, vol. 30, no. 6, pp. 6155–6163, 2025, doi: 10.1109/TMECH.2025.3552677.
[4]	D. Han, B. Mulyana, V. Stankovic, and S. Cheng, “A Survey on Deep Reinforcement Learning Algorithms for Robotic Manipulation,” Apr. 01, 2023, MDPI. doi: 10.3390/s23073762.
[5]	R. Liu, Y. Guo, R. Jin, and X. Zhang, “A Review of Natural-Language-Instructed Robot Execution Systems,” Sep. 01, 2024, Multidisciplinary Digital Publishing Institute (MDPI). doi: 10.3390/ai5030048.
[6]	X. Feng et al., “Natural Language Reinforcement Learning,” May 2025, [Online]. Available: http://arxiv.org/abs/2411.14251
[7]	I. Cleveston, A. C. Santana, P. D. P. Costa, R. R. Gudwin, A. S. Simões, and E. L. Colombini, “InstructRobot: A Model-Free Framework for Mapping Natural Language Instructions into Robot Motion,” Feb. 2025, [Online]. Available: http://arxiv.org/abs/2502.12861
[8]	S. G. Park, H. B. Kim, Y. J. Lee, W. J. Ahn, and M. T. Lim, “TARG: Tree of Action-reward Generation With Large Language Model for Cabinet Opening Using Manipulator,” Int. J. Control Autom. Syst., vol. 23, no. 2, pp. 449–458, Feb. 2025, doi: 10.1007/s12555-024-0528-6.
[9]	Y. Li, Q. Lyu, Y. Salam, J. Yang, and R. Ighil Nessouk, “Dual-LLM Hierarchical Task Planning and Skill Grounding for Mobile Manipulation in Long-Horizon Restroom Cleaning,” IEEE Access, vol. 13, pp. 190807–190819, 2025, doi: 10.1109/ACCESS.2025.3627503.
[10]	J. M. Dueñas Fajardo and P. F. Cárdenas Herrera, “Desarrollo de un modelo manipulador móvil basado en el robot móvil Robotino y con motores Dynamixel utilizando el framework de ROS,” 2020. doi: 10.13140/RG.2.2.25902.84808.
[11]	N. Sanghai and N. B. Brown, “Advances in Transformers for Robotic Applications: A Review,” Dec. 2024, [Online]. Available: http://arxiv.org/abs/2412.10599
[12]	A. Echchahed and P. S. Castro, “A Survey of State Representation Learning for Deep Reinforcement Learning,” Jun. 2025, [Online]. Available: http://arxiv.org/abs/2506.17518
[13]	X. Gong, D. Feng, K. Xu, B. Ding, and H. Wang, “Goal-Conditioned On-Policy Reinforcement Learning.”
[14]	C. Sun et al., “Fully Autonomous Real-World Reinforcement Learning with Applications to Mobile Manipulation,” Dec. 2021, [Online]. Available: http://arxiv.org/abs/2107.13545
[15]	C. Tang, B. Abbatematteo, J. Hu, R. Chandra, R. Martín-Martín, and P. Stone, “Deep Reinforcement Learning for Robotics: A Survey of Real-World Successes,” Robotics, and Autonomous Systems Downloaded from www.annualreviews.org. Guest, 2026, doi: 10.1146/annurev-control-030323.
[16]	G. Kwon, B. Kim, and N. K. Kwon, “Reinforcement Learning with Task Decomposition and Task-Specific Reward System for Automation of High-Level Tasks,” Biomimetics, vol. 9, no. 4, Apr. 2024, doi: 10.3390/biomimetics9040196.
[17]	D. Song et al., “Lucio at RoboCup@Home: An Open-Hardware Mobile Manipulator with Modular Software and On-Device LLM Planning,” in IEEE-RAS International Conference on Humanoid Robots, IEEE Computer Society, 2025, pp. 1019–1024. doi: 10.1109/Humanoids65713.2025.11203056.
[18]	Z. Zhang, F. Wurzberger, G. Schmid, S. Gottwald, and D. A. Braun, “Autonomous Learning From Success and Failure: Goal-Conditioned Supervised Learning with Negative Feedback,” Sep. 2025, [Online]. Available: http://arxiv.org/abs/2509.03206
[19]	R. Mendonca, E. Panov, B. Bucher, J. Wang, and D. Pathak, “Continuously Improving Mobile Manipulation with Autonomous Real-World RL,” Sep. 2024, [Online]. Available: http://arxiv.org/abs/2409.20568
[20]	K. Asuzu, H. Singh, and M. Idrissi, “Human–robot interaction through joint robot planning with large language models,” Intell. Serv. Robot., vol. 18, no. 2, pp. 261–277, Mar. 2025, doi: 10.1007/s11370-024-00570-1.
[21]	A. Brohan et al., “RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control,” Jul. 2023, [Online]. Available: http://arxiv.org/abs/2307.15818
[22]	P. Zheng, C. Li, J. Fan, and L. Wang, “A vision-language-guided and deep reinforcement learning-enabled approach for unstructured human-robot collaborative manufacturing task fulfilment,” CIRP Annals, vol. 73, no. 1, pp. 341–344, Jan. 2024, doi: 10.1016/j.cirp.2024.04.003.
[23]	Q. Sun et al., “Emma-X: An Embodied Multimodal Action Model with Grounded Chain of Thought and Look-ahead Spatial Reasoning,” Dec. 2024, [Online]. Available: http://arxiv.org/abs/2412.11974
[24]	Y. Tang, L. Zhang, S. Zhang, Y. Zhao, and X. Hao, “RoboAfford: A Dataset and Benchmark for Enhancing Object and Spatial Affordance Learning in Robot Manipulation,” in Proceedings of the 33rd ACM International Conference on Multimedia, New York, NY, USA: ACM, Oct. 2025, pp. 12706–12713. doi: 10.1145/3746027.3758209.
[25]	A. Mehri Shervedani, S. Li, N. Monaikul, B. Abbasi, M. Žefran, and B. Di Eugenio, “Multimodal Reinforcement Learning for Robots Collaborating with Humans,” Int. J. Soc. Robot., vol. 17, no. 12, pp. 3003–3025, Dec. 2025, doi: 10.1007/s12369-025-01287-6.
