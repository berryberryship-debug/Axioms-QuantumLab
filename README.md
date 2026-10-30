# Axioms-QuantumLab — Computational & Theoretical Quantum Physics

Bienvenue dans **Axioms-QuantumLab**, un laboratoire computationnel transversal dédié à la simulation et à l'étude de systèmes de physique quantique. Le dépôt couvre un large spectre, allant des qubits supraconducteurs (cQED) aux systèmes topologiques, en passant par le magnétisme et les systèmes quantiques ouverts.

## 🧭 Statut épistémique et méthodologie

Axioms-QuantumLab distingue explicitement trois niveaux de formalisation :

1. **Niveau physique** — Hamiltoniens, opérateurs, matrices de densité, observables, équations maîtresses et grandeurs dérivées. Les résultats sont interprétés dans le cadre des hypothèses physiques propres à chaque modèle.
2. **Niveau formel et computationnel** — Espaces de représentation, opérateurs de projection, variables latentes et architectures mathématiques destinées à explorer des correspondances entre différents formalismes.
3. **Niveau interprétatif et exploratoire** — Analogies conceptuelles, notamment celles reliant les structures physiques à des notions de cognition, de perception ou de conscience. Ces correspondances sont considérées comme des hypothèses ou des métaphores formelles, et non comme des résultats expérimentaux.

### Principes de validation

Les résultats numériques sont accompagnés, selon la nature du problème, de contrôles portant sur :
- la conservation des grandeurs physiques pertinentes (norme, énergie) ;
- la convergence numérique et la sensibilité aux paramètres ;
- les symétries et invariances attendues ;
- la stabilité des algorithmes ;
- la comparaison avec des solutions analytiques ou des références connues ;
- la reproductibilité des simulations.

*Règle d'or : La validation d'un calcul numérique ne doit jamais être confondue avec la validation de l'interprétation physique qu'on lui associe.*

---

## 📂 Architecture des Scripts

Les scripts sont organisés par domaine de recherche :

### 1. cQED & Qubits Supraconducteurs
*   `fluxonium_psi.py` : Modélisation d'un qubit Fluxonium. Calcul du spectre, des temps de cohérence (T1/T2), de la phase géométrique de Berry et de la métrique de Fubini-Study. *Note : Les mécanismes de décohérence (perte diélectrique, quasi-particules, bruit de flux) sont implémentés de manière phénoménologique, avec des hypothèses explicites dans le code.*
*   `transmon_lindblad.py` / `transmon_analyzer.py` : Analyse des décohérences et de la dynamique d'un qubit Transmon.
*   `run_transmon_spec.py` : Génération de spectres pour le Transmon.

### 2. Topologie & Matière Condensée
*   `ssh_kitaev_compare.py` : Étude comparative des modèles topologiques SSH et de la chaîne de Kitaev (Majorana).
*   `fix_hubbard.py` : Exploration du modèle de Hubbard.
*   `v99_skyrmion_2d.py` : Génération d'un champ de spins 2D (skyrmion) et calcul de la charge topologique. *Méthode : discrétisation par différences finies (gradients explicites dx/dy) sur grille avec conditions aux bords libres.*

### 3. Magnétisme & Ondes de Spin (Magnons)
*   `v100_magnon_dispersion.py` : Calcul de la relation de dispersion des magnons 1D pour un modèle de Heisenberg ferromagnétique.
*   `v101_magnon_packet.py` : Simulation dynamique spatio-temporelle d'un paquet d'ondes de spin via Transformée de Fourier Rapide (FFT). *Méthode : propagation spectrale exacte dans le modèle discret considéré.*

### 4. HPC & Systèmes Quantiques Ouverts
*   `moteur_ste50.py` : Moteur HPC (Numba) couplant hystérésis macroscopique et dynamique quantique. *Méthode : résolution par trajectoires quantiques (quantum trajectories / Monte Carlo wave-function) pour l'équation maîtresse de Lindblad via un estimateur de moyenne sur trajectoires parallélisées.* **(Génère des rapports DOCX)**.

### 5. Fondements & Formalismes Alternatifs
*   `bohm_hydro_mapping.py` : Exploration de la mécanique quantique de de Broglie-Bohm (interprétation hydrodynamique).
*   `run_hamiltonian.py` / `update_omega.py` / `scan_parameters.py` : Utilitaires de balayage paramétrique et d'évolution d'Hamiltoniens.

---

## 🔬 Détails des modules d'audit

### V99 — Champ de spins 2D et charge topologique
**[Observé numériquement]** Le script construit un ansatz analytique statique de type skyrmion. La densité topologique est calculée via la décomposition explicite des gradients spatiaux. La simulation donne une charge intégrée proche de \(Q \approx -1\).
**[Limite]** Existence mathématique d'une configuration non triviale. Pas de simulation de dynamique (LLG) ni de stabilisation physique (DMI).

### V100 — Dispersion des Magnons 1D
**[Démontré analytiquement]** Théorie linéaire des ondes de spin (Heisenberg). Formule \(E(q) = 2JS[1-\cos(qa)]\) et vitesse de groupe \(v_g = \frac{2JSa}{\hbar}\sin(qa)\).
**[Vérifié numériquement]** Pour les paramètres de référence (\(J=1.0\) meV, \(S=0.5\), \(a=1.0\) Å), la vitesse physique maximale calculée est de \(151.93\) m/s. La dispersion est quadratique près de \(\Gamma\) (\(E \propto q^2\)).
**[Limite]** Modèle à une particule (Holstein-Primakoff), sans interactions magnon-magnon ni amortissement.

### V101 — Dynamique et propagation d'un paquet d'ondes
**[Modèle et méthode]** Propagation spatio-temporelle d'un paquet gaussien par FFT. Propagation spectrale exacte dans le modèle discret considéré.
**[Vérifié numériquement]** Validation stricte sur 4 critères (conservation de la norme, conservation de l'énergie, vitesse du centroïde, étalement GVD).
**[Limites]** Centroïde linéaire valide uniquement sans traversée de la coupure périodique (vérifié par test de probabilité aux bords). Approx. des ondes de spin non-interagissantes.

---

## 🛡️ Validation numérique & Reproductibilité

Ce dépôt intègre des critères de validation pour distinguer les prototypes exploratoires des moteurs ayant passé un protocole de test. Chaque script documente ses invariants (ex : conservation de l'énergie ou de la norme) et ses tests de symétrie. Les paramètres de référence et les seuils de validation (ex : tolérances d'assertion) sont fixés dans le code pour garantir la reproductibilité des simulations.

> **⚠️ Note de transparence (Audit en cours) :** 
> La documentation ci-dessus reflète l'architecture et les intentions des scripts. Un audit croisé (Code vs README) est en cours pour s'assurer que chaque affirmation (mécanismes T1/T2, estimateurs Monte Carlo, conditions aux limites) correspond exactement à l'implémentation actuelle dans les fichiers `.py`. En cas de divergence, le code fait foi.

---

## 📌 Niveau de preuve

| Niveau | Signification |
|---|---|
| **[Démontré analytiquement]** | Propriété établie par dérivation mathématique |
| **[Vérifié numériquement]** | Propriété confirmée par des tests reproductibles |
| **[Modèle et méthode]** | Méthode définie pour un modèle physique donné |
| **[Observé numériquement]** | Résultat obtenu pour des paramètres et une discrétisation donnés |
| **[Hypothèse]** | Piste physique ou interprétation non validée |
| **[Limite]** | Restriction explicite de la portée d'un résultat |

---
*Développé sur Termux et VS Code. Architecture et validation scientifique itératives.*
