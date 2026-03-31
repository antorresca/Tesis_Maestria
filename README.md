# Tesis - Maestria en Automatizacion Industrial

## Autor: Ing. Andres Camilo Torres Cajamarca

---

## Arquitectura del Sistema

```mermaid
flowchart TD

    User([Usuario])

    subgraph PLN ["Módulo PLN ✓"]
        direction TB
        XLMR["Clasificador de intenciones\nXLM-RoBERTa fine-tuned\n94% F1 · 6 ms · ES+EN"]
        ENT["Extractor de entidades\nReglas / Regex\n{target, destination}"]
        XLMR --> ENT
    end

    subgraph TP ["Módulo Task Planning ▢"]
        direction TB
        DT["Árbol de decisiones\njerárquico\n{task_id, mode, objective}"]
    end

    subgraph DRL ["Módulo DRL ▢"]
        direction TB
        PPO["PPO vs SAC\n(comparar métricas)\nenv Gym"]
    end

    subgraph Robot ["Simulación · Docker ✓"]
        direction TB
        WBC["WBC / OSC\nROS Melodic · ~921 Hz\ncaja negra"]
        GAZ["Gazebo\nRobotino3 + INTERBOTIX VXSA-300"]
        WBC -->|Esfuerzos / Torques| GAZ
    end

    RI["robot_interface.py ✓\nroslibpy · WebSocket :9090"]

    User -->|Texto natural| PLN
    PLN -->|"{ intent, target, destination, confidence }"| TP
    TP -->|"{ task_id, mode, objective }"| DRL
    DRL -->|"/desired_traj"| RI
    RI <-->|WebSocket| WBC
    GAZ -->|"/mobile_manipulator/data\npose · joint_states"| DRL
```

### Estado de componentes

| Componente | Estado | Tecnología |
|---|---|---|
| Módulo PLN | **Completo ✓** | XLM-RoBERTa, 3 clases atómicas, 94% F1 |
| robot_interface.py | **Completo ✓** | roslibpy, rosbridge WebSocket |
| WBC + Gazebo + Docker | **Completo ✓** | ROS Melodic, ~921 Hz headless |
| Task Planning | Por desarrollar | Árbol de decisiones jerárquico |
| DRL | Por desarrollar | PPO vs SAC (Gymnasium) |
| Integración end-to-end | Por desarrollar | — |
