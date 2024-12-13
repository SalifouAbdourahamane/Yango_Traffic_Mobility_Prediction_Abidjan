# **Documentation: Prédiction de la Mobilité à Abidjan**

## **Table des Matières**

1. [Introduction](#introduction)  
2. [Description du Défi](#description-du-défi)  
3. [Exploration des Données](#exploration-des-données)  
4. [Ingénierie des Caractéristiques](#ingénierie-des-caractéristiques)  
5. [Architecture du Modèle](#architecture-du-modèle)  
6. [Approche Résiduelle avec LightGBM](#approche-résiduelle-avec-lightgbm)  
7. [Validation Croisée et Entraînement](#validation-croisée-et-entraînement)  
8. [Prédiction et Soumission](#prédiction-et-soumission)  
9. [Conclusion et Perspectives Futures](#conclusion-et-perspectives-futures)

---

## **Introduction**

Le but de ce projet est de développer un modèle prédictif utilisant l'apprentissage automatique pour estimer la **vitesse moyenne du trafic** sur les routes principales d’Abidjan, pendant les heures de pointe (matin et soir). Ces prédictions soutiendront l'optimisation des itinéraires pour Yango, une plateforme de transport, permettant des estimations précises des temps de trajet.

### **Importance**
Une prédiction précise des vitesses moyennes offre :  
- **Optimisation des ressources** : Réduction des temps de trajet et amélioration de l'expérience utilisateur.  
- **Impact environnemental** : Réduction des émissions grâce à des trajets optimisés.  
- **Planification dynamique** : Meilleure gestion des périodes de congestion.

---

## **Description du Défi**

### **Structure des Données**

Les données sont collectées par segments de route (`persistent_id`) et agrégées par **quart d'heure**. Les caractéristiques incluent :  
- **`speed_avg`** : Vitesse moyenne (cible).  
- **`count_norm`** : Nombre de trajets normalisé.  
- **Informations temporelles** : Heures, jours de semaine, et types de jour (ex. `first_weekday`).  

### **Objectif**
Prédire la vitesse moyenne des trajets pour :  
- **6h à 7h** (morning rush hour).  
- **18h à 19h** (evening rush hour).  

---

## **Exploration des Données**

### **1. Analyse Statistique**
- **Cycles temporels** : Les vitesses suivent des motifs quotidiens et hebdomadaires.  
- **Corrélations** : Les vitesses sont inversement corrélées au nombre de trajets (`count_norm`).  

### **2. Visualisation des Tendances**
- **Saisonnalité** : Les motifs de trafic suivent un comportement prévisible pendant les heures de pointe.  
- **Congestion** : Les heures de pointe affichent une utilisation élevée des routes, affectant la vitesse moyenne.

---

## **Ingénierie des Caractéristiques**

### **Pourquoi l'Ingénierie des Caractéristiques est Cruciale**
Les caractéristiques ajoutées permettent au modèle de capturer des dépendances complexes dans les données spatio-temporelles, telles que :  
- Les **tendances temporelles cycliques**.  
- Les **impacts de la congestion** sur les vitesses.  
- Les **interactions non linéaires** entre les variables.

### **Encodage des Variables Catégoriques**
Les colonnes catégoriques sont transformées en entiers via un **Label Encoding**, permettant à ces variables d’être interprétées par les modèles.

### **Standardisation**
Les colonnes numériques sont standardisées pour que toutes les variables aient une échelle similaire, ce qui est particulièrement important pour les modèles neuronaux.


### **Caractéristiques Créées**

#### **1. Lags Temporels**
- **Raison** : Le trafic est influencé par les vitesses des heures précédentes.  
- **Méthodologie** : Création de décalages temporels pour fournir un contexte historique.  
- **Caractéristiques** : 
  - **1 heure, 2 heures, 3 heures**.  
---

## **Architecture du Modèle**

## **1. Architecture du Réseau de Neurones : CNN-GRU avec Attention**

Le réseau de neurones constitue la base de cette solution, conçu pour modéliser efficacement les dynamiques temporelles et capturer des motifs complexes dans les données de trafic. Cette architecture est spécifiquement adaptée aux défis de la prévision du trafic  dense à Abidjan, notamment la nature cyclique des modèles de trafic, la forte variabilité pendant les heures de pointe et les dépendances temporelles.

### **Convolutional Neural Network (Conv1D)**
- Les couches Conv1D appliquent des convolutions dilatées pour extraire des caractéristiques temporelles localisées dans les données de trafic.
- Les dilations permettent au modèle d’« observer » sur de longues périodes sans augmenter la charge computationnelle. Cela est particulièrement pertinent pour capturer des motifs à court terme, comme des baisses soudaines de vitesse dues à des embouteillages ou à des événements météorologiques.
- Dans le contexte de la prévision du trafic, Conv1D aide à capturer les variations granulaires et fines dans les moyennes de vitesse sur des intervalles courts, tels que les segments de 15 minutes fournis dans les données.

### **Gated Recurrent Unit (GRU)**
- Les GRU sont spécialisées dans les données séquentielles et excellent à apprendre les dépendances à long terme en retenant le contexte historique pertinent tout en éliminant les informations non pertinentes.
- Dans cette tâche, les GRU permettent au modèle de se souvenir des motifs de trafic des heures ou même des jours précédents, ce qui est crucial pour prédire avec précision les vitesses pendant les heures de pointe qui suivent souvent un modèle cyclique.

### **Mécanisme d’Attention Multi-Têtes**
- Le mécanisme d’attention permet au modèle de prioriser dynamiquement certains points temporels dans la séquence qui sont les plus pertinents pour prédire la vitesse cible.
- Par exemple, pendant les heures de pointe du soir, les conditions de trafic plus tôt dans la journée ou la veille peuvent être plus influentes que des données historiques éloignées.
- L’attention multi-têtes améliore la capacité du modèle à capturer à la fois les dépendances temporelles à courte et longue portée, améliorant ainsi sa précision prédictive en se concentrant sur les principaux facteurs de variabilité du trafic.

### **Connexions Résiduelles**
- Les connexions résiduelles entre les couches convolutionnelles améliorent le flux du gradient pendant l’entraînement, atténuant le problème du gradient évanescent et permettant des architectures plus profondes.
- Ces connexions garantissent également que les caractéristiques extraites précédemment sont préservées pour les couches suivantes, conduisant à des représentations plus riches des données de trafic.

### **Couche de Sortie**
- Une couche dense produit la moyenne prédite des vitesses. Cette simple transformation linéaire garantit que le modèle se concentre sur l’optimisation de la métrique RMSE (Root Mean Squared Error), en alignement avec les critères d’évaluation du concours.

---

## **Avantages du Réseau de Neurones**
1. **Adaptabilité Temporelle** : En combinant Conv1D, GRU et Attention, le modèle capture à la fois les dépendances temporelles à court et long terme, cruciales pour la prévision du trafic.
2. **Focus Dynamique** : Les mécanismes d’attention permettent au modèle de se concentrer sur les intervalles temporels critiques, améliorant ainsi l’interprétabilité et la précision.
3. **Généralisation** : Les connexions résiduelles et les couches dropout aident à prévenir le surapprentissage, assurant une performance robuste sur des données non vues.

---

## **2. Apprentissage Résiduel avec LightGBM**

L’approche d’apprentissage résiduel est une stratégie avancée d’ensemble qui affine les prédictions du réseau neuronal. Voici comment cela fonctionne :

### **Idée Principale**
- Le réseau CNN-GRU fournit une prédiction initiale (`CNN_PRED`) de la vitesse moyenne du trafic. Bien que cette prédiction capture la plupart des motifs temporels et cycliques, elle peut échouer à modéliser certaines relations non linéaires ou à gérer des cas rares, tels que des changements soudains dans les conditions de trafic.
- LightGBM est utilisé pour apprendre à partir des erreurs résiduelles (`RESIDUAL = TRUE_VALUE - CNN_PRED`). Cela permet au modèle LightGBM de corriger les inexactitudes dans les prédictions CNN.

### **Pertinence pour le Défi**
- Les vitesses de trafic à Abidjan sont influencées par des facteurs hautement non linéaires tels qu’une augmentation soudaine de la densité des véhicules, des conditions routières localisées ou des événements imprévus.
- Le réseau neuronal peut ne pas capturer pleinement ces variations en raison de sa dépendance aux motifs temporels.
- LightGBM, grâce à son mécanisme d’amplification par gradient, excelle dans la modélisation d’interactions complexes et de dépendances non linéaires, ce qui en fait un choix idéal pour l’apprentissage résiduel.

---

### **Comment le Modèle Résiduel Améliore la Précision**
1. **Apprentissage Centré sur l’Erreur** :
   - LightGBM se concentre spécifiquement sur la correction des erreurs du réseau neuronal. Par exemple, si le CNN surestime la vitesse pendant un événement d’embouteillage, LightGBM peut réduire cette erreur en identifiant des motifs dans des caractéristiques comme `count_norm` (nombre de trajets) ou les valeurs décalées (`lagged speed values`).
2. **Interaction entre Caractéristiques** :
   - LightGBM peut exploiter les interactions entre caractéristiques que le réseau neuronal pourrait négliger, comme l’effet combiné entre l’utilisation PRB (utilisation réseau) et l’heure sur la vitesse du trafic.
3. **Réduction du Bruit** :
   - En isolant les résidus, LightGBM évite de reproduire les motifs déjà appris par le CNN et se concentre plutôt sur des relations secondaires subtiles dans les données.

---

## **3. Combinaison Finale des Prédictions**
La prédiction finale est une combinaison linéaire entre :
Prédiction Finale = {Prédiction CNN} + {Prédiction Résiduelle LightGBM} 

---

## **Validation Croisée et Entraînement**

### **1. Stratégie de Validation**
- **GroupKFold** : Garantit que les segments routiers (`persistent_id`) ne sont pas mélangés entre les ensembles d'entraînement et de validation.  

### **2. Processus d'Entraînement**
1. Division des données en 3 plis.  
2. Entraînement du CNN sur chaque pli.  
3. Calcul des résidus des prédictions CNN.  
4. Entraînement de LightGBM sur les résidus. 

### **Neural Network**:
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Epochs** : 20
- **Batch size**: 64
- **Early Stopping**: Stops training when validation loss plateaus.
- **Learning Rate Scheduler**: Reduces learning rate when improvement stalls.

### **LightGBM**

- **Learning Rate**: 0.05 
- **num_leaves**: 31 
- **feature_fraction**: 0.05
- **bagging_fraction'**: 0.8 
- **bagging_freq**: 5


### **Performance Finale:**
**Fold 1 :**
- CNN RMSE: 1.0882
- Final RMSE (CNN + LightGBM): 0.9851
**Fold 2 :**
- CNN RMSE: 1.1023
- Final RMSE (CNN + LightGBM): 1.0188

**Fold 3:**
- CNN RMSE: 1.0502
- Final RMSE (CNN + LightGBM): 0.9647

**RMSE on test data**:
- **0.1596**

- **Runtime**: **45 minutes on Google Colab T4 GPU**
 
---

## **Prédiction et Soumission**

### **1. Combinaison des Prédictions**
- Les prédictions finales combinent celles du CNN et de LightGBM.  
- **Formule** : `final_prediction = cnn_prediction + lgbm_residual_prediction`.  
### **2. Fichier de Soumission**
Format :  
```plaintext
ID                                       target
10001152554712362577_X_first_holiday_X_morning  15.4
20002381298761234178_X_last_weekday_X_evening   12.3
```
----
## **Conclusion et Perspectives Futures**
**Forces de la Solution**
- Approche hybride : Combine les points forts du CNN (dépendances séquentielles) et de LightGBM (relations non linéaires).
- Caractéristiques enrichies : Capturent des comportements complexes de trafic.
- Validation rigoureuse : Assure une bonne généralisation.

**Améliorations Futures :**
- Données Spatiales : Incorporer des relations entre les segments routiers.
- Données Externes : Ajouter des informations météorologiques ou des événements locaux.
- Optimisation : Affiner l'architecture CNN avec des techniques avancées comme LSTM.

Cette solution offre une solution robuste et optimale pour la prédiction des vitesses de trafic à Abidjan, soutenant les objectifs d'optimisation de Yango.




