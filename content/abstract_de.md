Mehr-Agenten-Reinforcement-Learning (MARL) ist ein aufkommendes Feld des
Reinforcement Learning mit realen Anwendungen wie unbemannte Luftfahrzeuge,
Such- und Rettungsdienste und Lagerorganisation. Es gibt viele verschiedene
Ansätze, um die Methoden des Single-Agent Reinforcement Learning auf MARL
anzuwenden. In dieser Arbeit geben wir einen Überblick über verschiedene
Lernmethoden und Umgebungseigenschaften und konzentrieren uns dann auf ein
Problem, das bei den meisten Varianten von MARL auftritt: Wie sollte ein Agent
die Informationen nutzen, die er aus einer großen und variierenden Anzahl von
Beobachtungen der Welt gesammelt hat, um Entscheidungen zu treffen? Wir
konzentrieren uns auf drei verschiedene Methoden zur Aggregation von
Beobachtungen und vergleichen sie hinsichtlich ihrer Trainingsleistung und
Sample Efficiency. Wir stellen eine auf Bayes'scher Konditionierung basierende
Policy-Architektur für die Aggregation vor und vergleichen sie mit der
Mittelwertaggregation und der Attention-Aggregation, die in verwandten Arbeiten
verwendet werden. Wir zeigen die Leistung der verschiedenen Methoden an einer
Reihe von kooperativen Aufgaben, die auf eine große Anzahl von Agenten skaliert
werden können, einschließlich Aufgaben, die andere Objekte in der Welt haben,
die von den Agenten beobachtet werden müssen, um die Aufgabe zu lösen.

Wir optimieren die Hyperparameter, um zeigen zu können, welche Parameter zu den
besten Ergebnissen für jede der Methoden führen. Darüber hinaus vergleichen wir
verschiedene Varianten der Bayes'schen Aggregation und vergleichen die kürzlich
eingeführte Lernmethode Trust Region Layers mit der allgemein bekannten Proximal
Policy Optimization (PPO).
