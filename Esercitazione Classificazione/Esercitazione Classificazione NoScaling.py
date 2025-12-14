import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE

# --- 1. Import dei Classificatori ---
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

print("--- 1. Caricamento Dati ---")
# Carica il dataset 'wine'
wine = load_wine()
# Crea un DataFrame pandas per manipolare i dati
# NOTA: Questa volta X sono i dati grezzi, non scalati
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

print(f"Dataset caricato: {X.shape[0]} campioni.")

# --- 2. Divisione Train/Test (STEP FONDAMENTALE) ---
# Dividiamo i dati grezzi PRIMA di qualsiasi altra operazione
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=0,  # Per la riproducibilità
    stratify=y  # Mantiene le proporzioni delle classi
)
print("--- 2. Dati divisi in Train e Test Set ---")


# --- 3. Funzione di Feature Selection ---
# Questa funzione sarà applicata SOLO ai dati di training
def select_features_by_correlation(df_train_with_target, target_col, threshold):
    """
    Seleziona feature basandosi sulla correlazione con il target
    calcolata SOLO sul set di training.
    """
    print(f"\n--- 3. Esecuzione Feature Selection (Soglia={threshold}) ---")
    corr = df_train_with_target.corr()
    corr_target = abs(corr[target_col])

    relevant_features = corr_target[corr_target > threshold]

    # Estrae i nomi delle feature (rimuovendo 'target' dalla lista)
    relevant_features_col = relevant_features.drop(target_col).keys().tolist()

    print(f"Feature selezionate: {len(relevant_features_col)}")
    # print(relevant_features_col)
    return relevant_features_col


# --- 3.1 Applicazione della Feature Selection ---
# 1. Crea un DataFrame di training temporaneo per calcolare la correlazione
X_train_temp = X_train.copy()
X_train_temp['target'] = np.array(y_train)

# 2. Trova le feature migliori (solo su X_train)
threshold = 0.45
selected_feature_names = select_features_by_correlation(
    X_train_temp,
    'target',
    threshold
)

# 3. Filtra X_train e X_test per tenere SOLO le feature selezionate
X_train_selected = X_train[selected_feature_names]
X_test_selected = X_test[selected_feature_names]

print(f"Dimensioni X_train dopo selezione: {X_train_selected.shape}")
print(f"Dimensioni X_test dopo selezione: {X_test_selected.shape}")

# --- 3.2 Plot della Heatmap di Correlazione ---
print("\n--- 3.2 Generazione Heatmap di Correlazione (su Train Set Grezzo) ---")
# Calcoliamo la matrice di correlazione (dati grezzi di train)
corr_matrix_raw = X_train_temp.corr()

plt.figure(figsize=(14, 12))
sns.heatmap(
    corr_matrix_raw,
    annot=True,
    fmt='.2f',
    cmap='GnBu',
    square=True,
    linewidths=.5,
    vmin=-1,
    vmax=1,
    center=0
)
plt.title("Matrice di Correlazione (Dati Grezzi di Training)", fontsize=14)
plt.savefig("heatmap_correlazione_grezza.png")  # Salva su file
plt.show()


# --- 4. Bilanciamento SMOTE (Come da Prof., ma Corretto) ---
# Applichiamo SMOTE SOLO al set di training
# Questo previene il data leakage sul test set
print("\n--- 4. Bilanciamento SMOTE (solo su Train Set) ---")
smote = SMOTE(random_state=0)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)

print(f"Dimensione y_train originale:\n{pd.Series(y_train).value_counts().sort_index()}")
print(f"Dimensione y_train dopo SMOTE:\n{pd.Series(y_train_resampled).value_counts().sort_index()}")

# --- 5. Definizione Modelli ---

models = {
    # --- Modelli di Riferimento ---
    "Random Forest": RandomForestClassifier(random_state=0),
    "Multinomial Naive Bayes": MultinomialNB(),

    # --- Confronto Decision Tree (variazione di max_depth) ---
    "Decision Tree (Depth=None)": DecisionTreeClassifier(random_state=0),
    "Decision Tree (Depth=3)": DecisionTreeClassifier(max_depth=3, random_state=0),
    "Decision Tree (Depth=5)": DecisionTreeClassifier(max_depth=5, random_state=0),
    "Decision Tree (Depth=8)": DecisionTreeClassifier(max_depth=8, random_state=0)
}

# --- 6. Addestramento e Confronto ---
results = {}
print("\n--- 6. Addestramento e Valutazione Modelli ---")

for model_name, model in models.items():
    print(f"\n--- Addestramento: {model_name} ---")

    # 1. Addestramento (sui dati di training bilanciati)
    model.fit(X_train_resampled, y_train_resampled)

    # 2. Predizione (sul test set originale, non bilanciato, non visto)
    y_pred = model.predict(X_test_selected)

    # 3. Valutazione
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = accuracy

    print(f"Accuracy: {accuracy * 100:.2f}%")

# --- 7. Riepilogo Finale ---
print("\n\n--- 7. RIEPILOGO CONFRONTO MODELLI ---")

sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)

print("Classifica finale dei modelli (per accuratezza):")
for model_name, accuracy in sorted_results:
    print(f"- {model_name}: {accuracy * 100:.2f}%")

# --- 8. Grafico per il modello migliore (Matrice di Confusione) ---
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

# --- 9. Grafico Finale (Istogramma per il Modello Migliore) ---
print(f"\n--- 9. Generazione Istogramma di Confronto per {best_model_name} ---")

# (Ri)calcoliamo le predizioni per il modello migliore
y_pred_best = best_model.predict(X_test_selected)

# Definiamo i "contenitori" per le classi 0, 1, 2
bins = [0, 1, 2, 3, 4, 5]
plt.hist(
    [y_pred_best, y_test],      # Confronta le previsioni migliori con i dati reali
    bins=bins,
    label=['y_pred (Previsioni)', 'y_test (Reali)'],
    align='left'
)

plt.xticks([0, 1, 2])
plt.title(f'Istogramma Classificazione - {best_model_name}')
plt.legend()
plt.savefig("istogramma_confronto_migliore.png")  # Salva su file
plt.show()


print("\n--- Esecuzione completata ---")