# --- 1. IMPORTAZIONE DELLE LIBRERIE ---

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine  # Per caricare il dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler    # Per la standardizzazione
from sklearn.metrics import accuracy_score # Per calcolare l'accuratezza
from sklearn.metrics import ConfusionMatrixDisplay  # Per plottare la matrice di confusione
from sklearn.tree import DecisionTreeClassifier # L'algoritmo di classificazione
from sklearn.ensemble import RandomForestClassifier # Modello "Ensemble" che usa molti alberi di decisione
from sklearn.neighbors import KNeighborsClassifier  # Modello basato sulla "vicinanza" dei campioni
from sklearn.naive_bayes import GaussianNB

# --- 2. CARICAMENTO E ANALISI ESPLORATIVA (EDA) ---

# Carica il dataset 'wine' dalla libreria sklearn
wine = load_wine()
# Crea un DataFrame pandas per manipolare i dati più facilmente
df = pd.DataFrame(wine.data, columns=wine.feature_names)

# Stampa le informazioni preliminari
print(f"Numero totale di campioni: {len(df)}")
print("\n--- Intestazione (prime 5 righe) ---")
print(df.head())
print("\n--- Statistiche Descrittive ---")
print(df.describe())
print("\n--- Controllo Valori Mancanti (per colonna) ---")
print(df.isnull().sum())  # Output: Non ci sono valori nulli

# --- 3. PREPARAZIONE DATI (PRE-PROCESSING) ---

# Definiamo le nostre feature (X) e il nostro target (y)
X = df
y = wine.target

# --- 3.1 Divisione Train/Test (STEP FONDAMENTALE) ---

# Dividiamo il dataset PRIMA di qualsiasi altra operazione (scaling, feature selection)
# per evitare il DATA LEAKAGE (contaminazione dei dati di test).
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.2,    # 20% dei dati sarà usato per il test set
    random_state = 0,   # Garantisce che lo split sia riproducibile
    stratify = y        # Mantiene la proporzione delle classi (0, 1, 2)
                        # sia nel train set che nel test set.
)

print(f"Dimensione Train Set: {X_train.shape}")
print(f"Dimensione Test Set: {X_test.shape}")

# --- 3.2 Standardizzazione (Scaling) ---

# Trasformiamo le feature per avere media 0 e deviazione standard 1
# Questo evita che feature con scale diverse (es. proline) dominino l'analisi.
scaler = StandardScaler()

# 1. IMPARA (fit) e TRASFORMA (transform) SOLO sul X_train
# Lo scaler impara la media e la std del training set e le applica.
X_train_scaled = scaler.fit_transform(X_train)

# 2. TRASFORMA (transform) X_test usando i parametri (media/std)
#    imparati da X_train. NON si usa .fit() o .fit_transform() qui!
X_test_scaled = scaler.transform(X_test)

# --- 4. ANALISI CORRELAZIONE E FEATURE SELECTION ---
# Questa analisi va fatta SOLO SUI DATI DI TRAINING

# Ricreiamo un DataFrame con i dati di training scalati
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=wine.feature_names)
# Aggiungiamo il target (y_train) per calcolare la correlazione con esso
X_train_scaled_df['target'] = np.array(y_train)

# Calcoliamo la matrice di correlazione completa
corr_matrix = X_train_scaled_df.corr()
# Estraiamo solo la correlazione di ogni feature CON IL TARGET
corr_target = abs(corr_matrix['target'])
# Ordiniamo e rimuoviamo 'target' stesso (che ha correlazione 1.0)
relevant_features = corr_target.drop('target').sort_values(ascending=False)
print("\n--- Matrice di Correlazione (Feature vs Target) ---")
print(relevant_features)

# --- 4.1 Plot della Heatmap di Correlazione ---

plt.figure(figsize=(14, 12))
sns.heatmap(
    corr_matrix,
    annot=True,         # Mostra i numeri (annotazioni)
    fmt='.2f',          # Formatta i numeri a 2 decimali
    cmap='GnBu',        # Colormap (Green-Blue) come nel PDF
    square=True,        # Celle quadrate
    linewidths=.5,      # Linee sottili tra le celle
    vmin=-1,            # Valore minimo della scala colori (-1)
    vmax=1,             # Valore massimo della scala colori (+1)
    center=0            # Centro della scala colori (per correlazioni)
)
plt.title("Matrice di correlazione delle feature", fontsize=14)
plt.savefig("heatmap_correlazione.png")
plt.show()

# --- 4.2 Funzione per la Feature Selection ---

def remove_unrelated_variability_with_target(df_model, target, threshold):
    """
    Seleziona le feature che hanno una correlazione (assoluta)
    con il 'target' superiore alla soglia 'threshold'.
    """
    corr = df_model.corr()
    corr_target = abs(corr[target])
    # Filtra le feature sopra la soglia
    relevant_features = corr_target[corr_target > threshold]

    print('\nFeatures correlation with target >', threshold)
    print(relevant_features.sort_values(ascending=False))

    # Estrae i nomi delle feature (rimuovendo 'target' dalla lista)
    relevant_features_col = relevant_features.drop(target).keys().tolist()

    print('\nFeatures correlated: ', len(relevant_features_col))
    print('Selected features:', relevant_features_col)

    return relevant_features_col

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=wine.feature_names)
X_train_scaled_df['target'] = np.array(y_train)

# Applichiamo la funzione (sempre e solo sui dati di training)
threshold = 0.45
selected_feature_names = remove_unrelated_variability_with_target(
    X_train_scaled_df,  # DataFrame di training con target
    'target',
    threshold
)

# --- 4.3 Applicazione della Selezione ---

# Ora filtriamo i nostri set (X_train e X_test) per tenere
# SOLO le feature selezionate.

# Ricreiamo i DataFrame scalati (questa volta senza target)
X_train_scaled_df_features = pd.DataFrame(X_train_scaled, columns=wine.feature_names)
X_test_scaled_df_features = pd.DataFrame(X_test_scaled, columns=wine.feature_names)

# Applichiamo il filtro
X_train_selected = X_train_scaled_df_features[selected_feature_names]
X_test_selected = X_test_scaled_df_features[selected_feature_names]

print(f"\nDimensioni X_train dopo: {X_train_selected.shape}")
print(f"Dimensioni X_test dopo: {X_test_selected.shape}")

# --- 5. DEFINIZIONE DEI MODELLI DA CONFRONTARE ---

models = {
    # --- Modelli di Riferimento ---
    "Random Forest": RandomForestClassifier(random_state=0),
    "Gaussian Naive Bayes": GaussianNB(),

    # --- Confronto Decision Tree (variazione di max_depth) ---
    "Decision Tree (Depth=None)": DecisionTreeClassifier(random_state=0),
    "Decision Tree (Depth=3)": DecisionTreeClassifier(max_depth=3, random_state=0),
    "Decision Tree (Depth=5)": DecisionTreeClassifier(max_depth=5, random_state=0),
    "Decision Tree (Depth=8)": DecisionTreeClassifier(max_depth=8, random_state=0),

    # --- Confronto KNN (variazione di k) ---
    "KNN (k=1)": KNeighborsClassifier(n_neighbors=1),
    "KNN (k=3)": KNeighborsClassifier(n_neighbors=3),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "KNN (k=11)": KNeighborsClassifier(n_neighbors=11)
}

# Creiamo un dizionario per salvare i risultati di ogni modello
results = {}

print("\n--- 6. ADDESTRAMENTO E VALUTAZIONE DEI MODELLI ---")

# Cicliamo su ogni modello definito sopra
for model_name, model in models.items():
    print(f"\n--- Addestramento: {model_name} ---")

    # 1. Addestramento
    # Addestriamo il modello sui dati di training (scalati e selezionati)
    model.fit(X_train_selected, y_train)

    # 2. Predizione
    # Usiamo il modello addestrato per predire sul test set
    y_pred = model.predict(X_test_selected)

    # 3. Valutazione
    # Calcoliamo l'accuratezza e la salviamo
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = accuracy

    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Stampiamo il report solo per riferimento
    # print(classification_report(y_test, y_pred, target_names=wine.target_names))

print("\n\n--- 7. RIEPILOGO CONFRONTO MODELLI ---")

# Ordiniamo i modelli dal migliore (accuratezza più alta) al peggiore
sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)

print("Classifica finale dei modelli (per accuratezza):")
for model_name, accuracy in sorted_results:
    print(f"- {model_name}: {accuracy * 100:.2f}%")

# --- Bonus: Mostriamo la Matrice di Confusione solo per il modello migliore ---
best_model_name, best_model_accuracy = sorted_results[0]
best_model = models[best_model_name]

print(f"\nMatrice di Confusione per il modello migliore ({best_model_name}):")
ConfusionMatrixDisplay.from_estimator(
    best_model,
    X_test_selected,
    y_test,
    display_labels=wine.target_names,
    cmap=plt.cm.Blues
)
plt.title(f"Matrice di Confusione - {best_model_name}")
plt.savefig(f"matrice_confusione_{best_model_name.replace(' ', '_')}.png")
plt.show()

print("\n--- Esecuzione completata ---")