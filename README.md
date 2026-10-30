# Axioms-QuantumLab — Computational & Theoretical Quantum Physics

Bienvenue dans **Axioms-QuantumLab**, un dépôt regroupant une collection de scripts et de moteurs de simulation Python dédiés à la physique quantique computationnelle. 

Ce laboratoire explore divers paradigmes de la physique moderne : des qubits supraconducteurs (cQED) aux systèmes topologiques, en passant par le magnétisme, les systèmes quantiques ouverts et les formalismes théoriques alternatifs.

## 🧭 Statut Épistémique

Une attention particulière est portée à la distinction entre :
1. **Les observables physiques** (Hamiltoniens, matrices de densité, phases géométriques, équations maîtresses).
2. **Les abstractions formelles** (les couches de projection comme "sensoriel", "langagier" ou "métacognition" dans certains scripts). 
3. **Les interprétations spéculatives** (toute analogie avec la conscience ou l'expérience subjective est explicitement documentée comme une métaphore formelle, non comme une preuve expérimentale).

## 📂 Architecture des Scripts

Les scripts sont organisés par domaine de recherche :

### 1. cQED & Qubits Supraconducteurs
*   `fluxonium_psi.py` : Modélisation complète d'un qubit Fluxonium. Calcul du spectre, des temps de cohérence (T1, T2), de la phase géométrique de Berry et de la métrique de Fubini-Study. Inclut une couche de projection formelle.
*   `transmon_lindblad.py` / `transmon_analyzer.py` : Analyse des décohérences et de la dynamique d'un qubit Transmon via l'équation de Lindblad.
*   `run_transmon_spec.py` : Génération de spectres pour le Transmon.

### 2. Topologie & Matière Condensée
*   `ssh_kitaev_compare.py` : Étude comparative des modèles topologiques SSH (Su-Schrieffer-Heeger) et de la chaîne de Kitaev (modèle de Majorana).
*   `fix_hubbard.py` : Exploration du modèle de Hubbard (interactions électroniques fortes).
*   `v99_skyrmion_2d.py` : Génération d'un champ de spins 2D (skyrmion) et calcul de sa charge topologique.

### 3. Magnétisme & Ondes de Spin (Magnons)
*   `v100_magnon_dispersion.py` : Calcul de la relation de dispersion des magnons 1D pour un modèle de Heisenberg ferromagnétique.
*   `v101_magnon_packet.py` : Simulation dynamique spatio-temporelle d'un paquet d'ondes de spin via Transformée de Fourier Rapide (FFT).

### 4. HPC & Systèmes Quantiques Ouverts
*   `moteur_ste50.py` : Moteur de calcul haute performance (HPC via Numba) couplant hystérésis macroscopique et dynamique quantique. Résout l'équation maîtresse de Lindblad via Monte Carlo parallélisé. **(Génère automatiquement des rapports scientifiques au format DOCX)**.

### 5. Fondements & Formalismes Alternatifs
*   `bohm_hydro_mapping.py` : Exploration de la mécanique quantique de de Broglie-Bohm (interprétation hydrodynamique).
*   `run_hamiltonian.py` / `update_omega.py` / `scan_parameters.py` : Utilitaires de balayage paramétrique et d'évolution d'Hamiltoniens génériques.

## 🛠️ Installation

Prérequis : Python 3.9 ou plus récent.

```bash
# Cloner le dépôt
git clone https://github.com/berryberryship-debug/Axioms-QuantumLab.git
cd Axioms-QuantumLab

# Installer les dépendances
pip install -r requirements.txt
